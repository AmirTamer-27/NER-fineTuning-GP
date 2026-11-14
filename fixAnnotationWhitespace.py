"""Normalize whitespace at positive entity-span boundaries.

This script updates the active bilingual JSONL splits atomically. It never
changes document text or ignore_spans. Every adjusted or removed positive span
is written to a reversible JSONL audit log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "finaira_bilingual_ner_v2"
SPLITS = ("train", "validation", "test")
NEW_VERSION = "finaira-bilingual-ner 2.1"
AUDIT_FILENAME = "annotation_whitespace_cleanup.jsonl"
REPORT_FILENAME = "annotation_whitespace_cleanup_report.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trim whitespace from positive entity annotations."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated rewrite. Without this flag, only scan.",
    )
    parser.add_argument(
        "--preserve-version",
        action="store_true",
        help="Keep the version declared in current_manifest.json.",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_bounds(text, start, end):
    new_start = start
    new_end = end
    while new_start < new_end and text[new_start].isspace():
        new_start += 1
    while new_end > new_start and text[new_end - 1].isspace():
        new_end -= 1
    return new_start, new_end


def validate_original_span(document, span, split, index):
    text = document["text"]
    start, end = span["start"], span["end"]
    if not 0 <= start < end <= len(text):
        raise ValueError(
            f"Invalid {split} span in {document['id']} at index {index}: {span}"
        )
    if "text" in span and text[start:end] != span["text"]:
        raise ValueError(
            f"Text mismatch in {split}/{document['id']} at span {index}: {span}"
        )


def scan(data_dir):
    statistics = Counter()
    by_split = defaultdict(Counter)
    by_label = Counter()
    by_source = Counter()
    document_counts = Counter()

    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = json.loads(line)
                document_counts[split] += 1
                for index, span in enumerate(document.get("spans", [])):
                    validate_original_span(document, span, split, index)
                    start, end = span["start"], span["end"]
                    new_start, new_end = normalized_bounds(
                        document["text"], start, end
                    )
                    if (start, end) == (new_start, new_end):
                        continue
                    removed = new_start == new_end
                    left = new_start - start
                    right = end - new_end
                    statistics["affected_spans"] += 1
                    statistics["removed_all_whitespace_spans"] += int(removed)
                    statistics["trimmed_spans"] += int(not removed)
                    statistics["left_characters_removed"] += left
                    statistics["right_characters_removed"] += right
                    statistics["spans_with_left_whitespace"] += int(left > 0)
                    statistics["spans_with_right_whitespace"] += int(right > 0)
                    by_split[split]["affected_spans"] += 1
                    by_split[split]["removed_all_whitespace_spans"] += int(removed)
                    by_split[split]["trimmed_spans"] += int(not removed)
                    by_label[span["label"]] += 1
                    by_source[span.get("source", "unknown")] += 1

                # Ignore regions are intentionally not modified, but validate
                # their stored character offsets before any rewrite.
                for index, span in enumerate(document.get("ignore_spans", [])):
                    validate_original_span(document, span, split, index)

    for key in (
        "affected_spans",
        "removed_all_whitespace_spans",
        "trimmed_spans",
        "left_characters_removed",
        "right_characters_removed",
        "spans_with_left_whitespace",
        "spans_with_right_whitespace",
    ):
        statistics.setdefault(key, 0)
    for split in SPLITS:
        for key in (
            "affected_spans",
            "removed_all_whitespace_spans",
            "trimmed_spans",
        ):
            by_split[split].setdefault(key, 0)

    return {
        "summary": dict(statistics),
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_label": dict(sorted(by_label.items())),
        "by_source": dict(sorted(by_source.items())),
        "documents": dict(document_counts),
    }


def temporary_path(directory, prefix):
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=directory)
    os.close(descriptor)
    return Path(raw_path)


def rewrite_splits(data_dir, expected_scan, record_version):
    temp_splits = {}
    audit_temp = temporary_path(data_dir, ".annotation-whitespace-audit-")
    actual = Counter()

    try:
        with audit_temp.open("w", encoding="utf-8") as audit_handle:
            for split in SPLITS:
                source_path = data_dir / f"{split}.jsonl"
                temp_path = temporary_path(data_dir, f".{split}-whitespace-")
                temp_splits[split] = temp_path
                with source_path.open(encoding="utf-8") as source, temp_path.open(
                    "w", encoding="utf-8"
                ) as target:
                    for line_number, line in enumerate(source, start=1):
                        if not line.strip():
                            continue
                        document = json.loads(line)
                        original_text = document["text"]
                        normalized_spans = []

                        for span_index, span in enumerate(document.get("spans", [])):
                            start, end = span["start"], span["end"]
                            new_start, new_end = normalized_bounds(
                                original_text, start, end
                            )
                            if (start, end) == (new_start, new_end):
                                normalized_spans.append(span)
                                continue

                            removed = new_start == new_end
                            before = span.copy()
                            after = None
                            if not removed:
                                after = span.copy()
                                after["start"] = new_start
                                after["end"] = new_end
                                after["text"] = original_text[new_start:new_end]
                                normalized_spans.append(after)

                            audit_record = {
                                "split": split,
                                "document_id": document["id"],
                                "line_number": line_number,
                                "original_span_index": span_index,
                                "action": (
                                    "removed_all_whitespace" if removed else "trimmed"
                                ),
                                "before": before,
                                "after": after,
                                "left_characters_removed": new_start - start,
                                "right_characters_removed": end - new_end,
                            }
                            audit_handle.write(
                                json.dumps(audit_record, ensure_ascii=False) + "\n"
                            )
                            actual["affected_spans"] += 1
                            actual["removed_all_whitespace_spans"] += int(removed)
                            actual["trimmed_spans"] += int(not removed)

                        document["spans"] = normalized_spans
                        document["version"] = record_version
                        if document["text"] != original_text:
                            raise AssertionError(
                                f"Document text changed unexpectedly: {document['id']}"
                            )
                        target.write(json.dumps(document, ensure_ascii=False) + "\n")

        for key in (
            "affected_spans",
            "removed_all_whitespace_spans",
            "trimmed_spans",
        ):
            if actual[key] != expected_scan["summary"].get(key, 0):
                raise AssertionError(
                    f"Rewrite count for {key} was {actual[key]}, expected "
                    f"{expected_scan['summary'].get(key, 0)}"
                )
        return temp_splits, audit_temp
    except Exception:
        for path in temp_splits.values():
            path.unlink(missing_ok=True)
        audit_temp.unlink(missing_ok=True)
        raise


def validate_rewrite(temp_splits, expected_documents, record_version):
    document_counts = Counter()
    span_counts = Counter()
    seen_ids = set()

    for split in SPLITS:
        with temp_splits[split].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = json.loads(line)
                document_counts[split] += 1
                if document["id"] in seen_ids:
                    raise ValueError(f"Duplicate document ID: {document['id']}")
                seen_ids.add(document["id"])
                if document.get("version") != record_version:
                    raise ValueError(f"Unexpected version in {document['id']}")

                ordered = sorted(
                    document.get("spans", []), key=lambda span: span["start"]
                )
                for index, span in enumerate(ordered):
                    validate_original_span(document, span, split, index)
                    value = document["text"][span["start"] : span["end"]]
                    if value != value.strip():
                        raise ValueError(
                            f"Whitespace remains in {split}/{document['id']}: {span}"
                        )
                    span_counts[split] += 1
                for left, right in zip(ordered, ordered[1:]):
                    if right["start"] < left["end"]:
                        raise ValueError(
                            f"Overlapping spans in {split}/{document['id']}"
                        )

    if dict(document_counts) != expected_documents:
        raise AssertionError(
            f"Document counts changed: {dict(document_counts)} != {expected_documents}"
        )
    return dict(document_counts), dict(span_counts)


def write_json_atomic(path, value):
    temp_path = temporary_path(path.parent, f".{path.name}-")
    try:
        temp_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def update_metadata(data_dir, report, hashes, total_entity_spans, record_version):
    cleanup_entry = {
        "applied_on": date.today().isoformat(),
        "version": record_version,
        "field_modified": "spans",
        "document_text_modified": False,
        "ignore_spans_modified": False,
        "policy": (
            "Trim Unicode whitespace from positive entity boundaries; remove "
            "positive annotations consisting entirely of whitespace."
        ),
        **report,
        "total_entity_spans_after": total_entity_spans,
        "data_sha256": hashes,
        "audit": AUDIT_FILENAME,
        "script": str(Path(__file__).resolve()),
    }

    manifest_path = data_dir / "current_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = record_version
    manifest["total_entity_spans"] = total_entity_spans
    manifest["data_sha256"] = hashes
    manifest["annotation_whitespace_cleanup"] = cleanup_entry
    write_json_atomic(manifest_path, manifest)

    verification_path = data_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    verification["passed"] = True
    verification["total_entity_spans"] = total_entity_spans
    verification["data_sha256"] = hashes
    verification["annotation_whitespace_check"] = {
        "passed": True,
        "remaining_positive_spans_with_boundary_whitespace": 0,
        "document_text_unchanged": True,
        "ignore_spans_unchanged": True,
        "cleanup": cleanup_entry,
    }
    write_json_atomic(verification_path, verification)

    cleanup_path = data_dir / "cleanup_report.json"
    cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
    cleanup["annotation_whitespace_cleanup"] = cleanup_entry
    write_json_atomic(cleanup_path, cleanup)

    write_json_atomic(data_dir / REPORT_FILENAME, cleanup_entry)


def main():
    args = parse_args()
    for split in SPLITS:
        if not (args.data_dir / f"{split}.jsonl").is_file():
            raise FileNotFoundError(args.data_dir / f"{split}.jsonl")

    manifest = json.loads(
        (args.data_dir / "current_manifest.json").read_text(encoding="utf-8")
    )
    record_version = manifest["version"] if args.preserve_version else NEW_VERSION

    report = scan(args.data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not args.apply:
        print("Scan only; rerun with --apply to update the split files.", flush=True)
        return 0

    temp_splits, audit_temp = rewrite_splits(
        args.data_dir, report, record_version
    )
    try:
        document_counts, span_counts = validate_rewrite(
            temp_splits, report["documents"], record_version
        )
        hashes = {
            f"{split}.jsonl": sha256(temp_splits[split]) for split in SPLITS
        }
        total_entity_spans = sum(span_counts.values())

        for split in SPLITS:
            os.replace(temp_splits[split], args.data_dir / f"{split}.jsonl")
        os.replace(audit_temp, args.data_dir / AUDIT_FILENAME)

        report["documents"] = document_counts
        report["entity_spans_after_by_split"] = span_counts
        update_metadata(
            args.data_dir, report, hashes, total_entity_spans, record_version
        )
    finally:
        for path in temp_splits.values():
            path.unlink(missing_ok=True)
        audit_temp.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "applied": True,
                "version": record_version,
                "total_entity_spans": total_entity_spans,
                "data_sha256": hashes,
                "audit": str((args.data_dir / AUDIT_FILENAME).resolve()),
                "report": str((args.data_dir / REPORT_FILENAME).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
