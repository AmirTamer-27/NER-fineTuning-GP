"""Round-trip annotated spans through the production BIO preprocessor.

The check uses ``prepare_ner_dataset`` to tokenize text and create BIO labels,
then decodes those labels back to document-level character spans. It does not
load the XLM-R encoder and it does not evaluate model predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_PACKAGES = ROOT / ".finaira-packages"
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))
os.environ.setdefault("HF_HOME", str(ROOT / ".huggingface"))

from transformers import AutoTokenizer

from prepare_data import prepare_ner_dataset


DEFAULT_DATA = ROOT / "finaira_bilingual_ner_v2" / "train.jsonl"
DEFAULT_OUTPUT = ROOT / "decoder_check_results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check span -> BIO -> span round-trip fidelity."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--sample-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Return exit code 1 when the round trip is not exact.",
    )
    return parser.parse_args()


def span_key(span):
    return span["start"], span["end"], span["label"]


def span_record(text, key):
    start, end, label = key
    return {"start": start, "end": end, "label": label, "text": text[start:end]}


def overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def inspect_dataset(path):
    """Count documents and collect all entity types without retaining text."""
    document_count = 0
    entity_types = set()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            document_count += 1
            entity_types.update(span["label"] for span in record.get("spans", []))
    return document_count, sorted(entity_types), digest.hexdigest()


def selected_documents(path, selected_indices):
    """Stream only the deterministically selected documents from JSONL."""
    wanted = iter(sorted(selected_indices))
    next_index = next(wanted, None)
    if next_index is None:
        return
    document_index = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if document_index == next_index:
                yield document_index, json.loads(line)
                next_index = next(wanted, None)
                if next_index is None:
                    return
            document_index += 1


def decode_bio_examples(examples, id2label, text):
    """Decode chunk BIO labels and deduplicate spans created by stride overlap."""
    decoded_occurrences = []
    invalid_i = []

    for example in examples:
        current = None

        def close_current():
            nonlocal current
            if current is not None:
                decoded_occurrences.append(tuple(current))
                current = None

        for label_id, offset in zip(example["labels"], example["offset_mapping"]):
            start, end = offset
            if label_id == -100 or start == end:
                close_current()
                continue

            tag = id2label[label_id]
            if tag == "O":
                close_current()
                continue

            prefix, entity_type = tag.split("-", 1)
            if prefix == "B":
                close_current()
                current = [start, end, entity_type]
            elif prefix == "I" and current is not None and current[2] == entity_type:
                current[1] = max(current[1], end)
            elif prefix == "I":
                # A bare I tag is invalid BIO. Recover it as a new span so the
                # report can show the resulting mismatch instead of hiding it.
                close_current()
                current = [start, end, entity_type]
                invalid_i.append({
                    "document_id": example["document_id"],
                    "chunk_index": example["chunk_index"],
                    "offset": [start, end],
                    "tag": tag,
                    "text": text[start:end],
                })
            else:
                raise ValueError(f"Unsupported BIO tag: {tag}")
        close_current()

    occurrence_counts = Counter(decoded_occurrences)
    return set(occurrence_counts), occurrence_counts, invalid_i


def ignored_gold_spans(document):
    ignored_regions = document.get("ignore_spans", [])
    return {
        span_key(span)
        for span in document.get("spans", [])
        if any(
            span["start"] < region["end"] and region["start"] < span["end"]
            for region in ignored_regions
        )
    }


def boundary_candidates(missing, extra, text):
    candidates = []
    for missing_key in sorted(missing):
        possible = [
            extra_key
            for extra_key in extra
            if missing_key[2] == extra_key[2]
            and overlaps(missing_key[:2], extra_key[:2])
        ]
        if possible:
            best = max(
                possible,
                key=lambda key: min(missing_key[1], key[1])
                - max(missing_key[0], key[0]),
            )
            candidates.append({
                "gold": span_record(text, missing_key),
                "decoded": span_record(text, best),
            })
    return candidates


def increment(counter, key, amount=1):
    counter[key] += amount


def main():
    args = parse_args()
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample-fraction must be in the interval (0, 1].")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    document_count, entity_types, data_sha256 = inspect_dataset(args.data)
    sample_size = round(document_count * args.sample_fraction)
    sample_size = max(1, min(document_count, sample_size))
    selected_indices = set(random.Random(args.seed).sample(range(document_count), sample_size))

    print(
        f"Loading tokenizer {args.checkpoint!r}; checking {sample_size:,} of "
        f"{document_count:,} documents ({sample_size / document_count:.2%}).",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("The round-trip check requires a fast tokenizer.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mismatches_path = args.output_dir / "decoder_check_mismatches.jsonl"
    sample_ids_path = args.output_dir / "decoder_check_sample_ids.txt"
    summary_path = args.output_dir / "decoder_check_summary.json"

    totals = Counter()
    per_language = defaultdict(Counter)
    per_label = defaultdict(Counter)
    sample_ids = []

    with mismatches_path.open("w", encoding="utf-8") as mismatch_file:
        for processed, (_, document) in enumerate(
            selected_documents(args.data, selected_indices), start=1
        ):
            document_id = document["id"]
            language = document.get("language", "unknown")
            text = document["text"]
            sample_ids.append(document_id)
            gold = {span_key(span) for span in document.get("spans", [])}
            ignored = ignored_gold_spans(document)
            expected = gold - ignored
            processing_error = None
            invalid_i = []
            decoded = set()
            occurrence_counts = Counter()
            unsupervised = []

            try:
                examples, label2id, coverage = prepare_ner_dataset(
                    [document],
                    tokenizer,
                    entity_types,
                    max_length=args.max_length,
                    stride=args.stride,
                    include_metadata=True,
                )
                id2label = {value: key for key, value in label2id.items()}
                decoded, occurrence_counts, invalid_i = decode_bio_examples(
                    examples, id2label, text
                )
                unsupervised = coverage[0]["unsupervised_entities"]
            except Exception as error:  # retain all bad documents in one report
                processing_error = f"{type(error).__name__}: {error}"

            matched = expected & decoded
            missing = expected - decoded
            extra = decoded - expected
            perfect = not processing_error and not missing and not extra and not invalid_i

            increment(totals, "documents")
            increment(totals, "perfect_documents", int(perfect))
            increment(totals, "processing_errors", int(processing_error is not None))
            increment(totals, "gold_spans", len(gold))
            increment(totals, "expected_spans", len(expected))
            increment(totals, "ignored_spans", len(ignored))
            increment(totals, "decoded_unique_spans", len(decoded))
            increment(totals, "decoded_occurrences", sum(occurrence_counts.values()))
            increment(totals, "matched_spans", len(matched))
            increment(totals, "missing_spans", len(missing))
            increment(totals, "extra_spans", len(extra))
            increment(totals, "invalid_i_tags", len(invalid_i))
            increment(totals, "unsupervised_spans", len(unsupervised))

            language_stats = per_language[language]
            increment(language_stats, "documents")
            increment(language_stats, "perfect_documents", int(perfect))
            increment(language_stats, "processing_errors", int(processing_error is not None))
            increment(language_stats, "expected_spans", len(expected))
            increment(language_stats, "matched_spans", len(matched))
            increment(language_stats, "missing_spans", len(missing))
            increment(language_stats, "extra_spans", len(extra))

            for key in expected:
                increment(per_label[key[2]], "expected")
            for key in matched:
                increment(per_label[key[2]], "matched")
            for key in missing:
                increment(per_label[key[2]], "missing")
            for key in extra:
                increment(per_label[key[2]], "extra")

            if not perfect or ignored or unsupervised:
                result = {
                    "document_id": document_id,
                    "language": language,
                    "processing_error": processing_error,
                    "gold_span_count": len(gold),
                    "expected_span_count": len(expected),
                    "decoded_span_count": len(decoded),
                    "missing_spans": [span_record(text, key) for key in sorted(missing)],
                    "extra_spans": [span_record(text, key) for key in sorted(extra)],
                    "ignored_spans": [span_record(text, key) for key in sorted(ignored)],
                    "unsupervised_spans": unsupervised,
                    "invalid_i_tags": invalid_i,
                    "boundary_candidates": boundary_candidates(missing, extra, text),
                }
                mismatch_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            if processed % 1000 == 0 or processed == sample_size:
                print(
                    f"Processed {processed:,}/{sample_size:,}; exact documents: "
                    f"{totals['perfect_documents']:,}; missing spans: "
                    f"{totals['missing_spans']:,}; extra spans: {totals['extra_spans']:,}",
                    flush=True,
                )

    precision_denominator = totals["matched_spans"] + totals["extra_spans"]
    recall_denominator = totals["expected_spans"]
    summary = {
        "passed": (
            totals["processing_errors"] == 0
            and totals["missing_spans"] == 0
            and totals["extra_spans"] == 0
            and totals["invalid_i_tags"] == 0
            and totals["unsupervised_spans"] == 0
        ),
        "scope": {
            "data": str(args.data.resolve()),
            "data_sha256": data_sha256,
            "total_documents": document_count,
            "sample_fraction_requested": args.sample_fraction,
            "sample_documents": sample_size,
            "sample_fraction_actual": sample_size / document_count,
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "max_length": args.max_length,
            "stride": args.stride,
            "entity_types": entity_types,
        },
        "totals": dict(totals),
        "exact_span_precision": (
            totals["matched_spans"] / precision_denominator
            if precision_denominator
            else 1.0
        ),
        "exact_span_recall": (
            totals["matched_spans"] / recall_denominator
            if recall_denominator
            else 1.0
        ),
        "perfect_document_rate": totals["perfect_documents"] / totals["documents"],
        "per_language": {key: dict(value) for key, value in sorted(per_language.items())},
        "per_label": {key: dict(value) for key, value in sorted(per_label.items())},
        "artifacts": {
            "mismatches": str(mismatches_path.resolve()),
            "sample_ids": str(sample_ids_path.resolve()),
        },
        "notes": [
            "Expected spans exclude annotations intersecting ignore_spans because the production preprocessor assigns their tokens -100.",
            "Decoded duplicates from overlapping stride windows are deduplicated by exact document start, end, and label.",
            "This validates gold BIO alignment and decoding, not model prediction quality.",
        ],
    }
    sample_ids_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "passed": summary["passed"],
        "documents": totals["documents"],
        "perfect_documents": totals["perfect_documents"],
        "processing_errors": totals["processing_errors"],
        "expected_spans": totals["expected_spans"],
        "matched_spans": totals["matched_spans"],
        "missing_spans": totals["missing_spans"],
        "extra_spans": totals["extra_spans"],
        "ignored_spans": totals["ignored_spans"],
        "summary": str(summary_path),
    }, indent=2), flush=True)

    return 1 if args.fail_on_mismatch and not summary["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
