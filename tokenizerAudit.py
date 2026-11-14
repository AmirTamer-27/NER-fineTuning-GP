"""Run the production tokenizer/alignment policy over complete JSONL splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


DEFAULT_DATA = ROOT / "finaira_bilingual_ner_v2"
DEFAULT_OUTPUT = ROOT / "tokenizer_audit_results"
SPLITS = ("train", "validation", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit XLM-R BIO alignment over complete dataset splits."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=2000)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_splits(data_dir):
    document_counts = Counter()
    entity_types = set()
    hashes = {}
    for split in SPLITS:
        path = data_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                document = json.loads(line)
                document_counts[split] += 1
                entity_types.update(
                    span["label"] for span in document.get("spans", [])
                )
        hashes[path.name] = sha256(path)
    return dict(document_counts), sorted(entity_types), hashes


def increment_label(counter, records):
    for record in records:
        counter[record["entity"]["label"]] += 1


def main():
    args = parse_args()
    document_counts, entity_types, hashes = inspect_splits(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset mappings.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "tokenizer_alignment_cases.jsonl"
    errors_path = args.output_dir / "tokenizer_processing_errors.jsonl"
    summary_path = args.output_dir / "tokenizer_audit_summary.json"

    totals = Counter()
    by_split = defaultdict(Counter)
    by_language = defaultdict(Counter)
    punctuation_by_label = Counter()
    ambiguous_by_label = Counter()
    ambiguous_by_reason = Counter()
    outside_text_counts = Counter()
    non_alphanumeric_annotations_by_label = Counter()

    with details_path.open("w", encoding="utf-8") as details, errors_path.open(
        "w", encoding="utf-8"
    ) as errors:
        for split in SPLITS:
            path = args.data_dir / f"{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for split_index, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    document = json.loads(line)
                    language = document.get("language") or "unknown"
                    for span in document.get("spans", []):
                        value = document["text"][span["start"]:span["end"]]
                        if not any(character.isalnum() for character in value):
                            non_alphanumeric_annotations_by_label[
                                span["label"]
                            ] += 1
                    totals["documents"] += 1
                    by_split[split]["documents"] += 1
                    by_language[language]["documents"] += 1
                    try:
                        _, _, coverage = prepare_ner_dataset(
                            [document],
                            tokenizer,
                            entity_types,
                            max_length=args.max_length,
                            stride=args.stride,
                        )
                        report = coverage[0]
                    except Exception as error:
                        totals["processing_errors"] += 1
                        by_split[split]["processing_errors"] += 1
                        by_language[language]["processing_errors"] += 1
                        errors.write(json.dumps({
                            "split": split,
                            "document_id": document.get("id"),
                            "error": f"{type(error).__name__}: {error}",
                        }, ensure_ascii=False) + "\n")
                        continue

                    punctuation = report["punctuation_boundary_conflicts"]
                    ambiguous = report["ambiguous_boundary_conflicts"]
                    ambiguous_entities = report["ambiguous_entities"]
                    unsupervised = report["unsupervised_entities"]
                    ambiguous_token_offsets = {
                        (record["token_start"], record["token_end"])
                        for record in ambiguous
                    }
                    multi_entity_token_offsets = {
                        (record["token_start"], record["token_end"])
                        for record in ambiguous
                        if record["reason"] == "token_overlaps_multiple_entities"
                    }
                    punctuation_entities = {
                        (
                            record["entity"]["start"],
                            record["entity"]["end"],
                            record["entity"]["label"],
                        )
                        for record in punctuation
                    }
                    totals["target_entities"] += report["target_entities"]
                    totals["supervised_entities"] += report["supervised_entities"]
                    totals["ignored_entities"] += report["ignored_entities"]
                    totals["punctuation_tokens"] += len(punctuation)
                    totals["punctuation_entities"] += len(punctuation_entities)
                    totals["ambiguous_tokens"] += len(ambiguous_token_offsets)
                    totals["ambiguous_token_entity_conflicts"] += len(ambiguous)
                    totals["multi_entity_tokens"] += len(multi_entity_token_offsets)
                    totals["ambiguous_entities"] += len(ambiguous_entities)
                    totals["unsupervised_entities"] += len(unsupervised)
                    by_split[split]["punctuation_tokens"] += len(punctuation)
                    by_split[split]["punctuation_entities"] += len(
                        punctuation_entities
                    )
                    by_split[split]["ambiguous_tokens"] += len(
                        ambiguous_token_offsets
                    )
                    by_split[split]["ambiguous_token_entity_conflicts"] += len(
                        ambiguous
                    )
                    by_split[split]["multi_entity_tokens"] += len(
                        multi_entity_token_offsets
                    )
                    by_split[split]["ambiguous_entities"] += len(ambiguous_entities)
                    by_split[split]["unsupervised_entities"] += len(unsupervised)
                    by_language[language]["target_entities"] += report[
                        "target_entities"
                    ]
                    by_language[language]["supervised_entities"] += report[
                        "supervised_entities"
                    ]
                    by_language[language]["punctuation_tokens"] += len(punctuation)
                    by_language[language]["punctuation_entities"] += len(
                        punctuation_entities
                    )
                    by_language[language]["ambiguous_tokens"] += len(
                        ambiguous_token_offsets
                    )
                    by_language[language]["ambiguous_token_entity_conflicts"] += len(
                        ambiguous
                    )
                    by_language[language]["multi_entity_tokens"] += len(
                        multi_entity_token_offsets
                    )
                    by_language[language]["ambiguous_entities"] += len(
                        ambiguous_entities
                    )
                    by_language[language]["unsupervised_entities"] += len(
                        unsupervised
                    )
                    if punctuation:
                        totals["documents_with_punctuation"] += 1
                        by_split[split]["documents_with_punctuation"] += 1
                        by_language[language]["documents_with_punctuation"] += 1
                    if ambiguous:
                        totals["documents_with_ambiguous"] += 1
                        by_split[split]["documents_with_ambiguous"] += 1
                        by_language[language]["documents_with_ambiguous"] += 1
                    if unsupervised:
                        totals["documents_with_unsupervised"] += 1
                        by_split[split]["documents_with_unsupervised"] += 1

                    increment_label(punctuation_by_label, punctuation)
                    increment_label(ambiguous_by_label, ambiguous)
                    for record in punctuation:
                        outside_text_counts[record["outside_text"]] += 1
                    for record in ambiguous:
                        ambiguous_by_reason[record["reason"]] += 1

                    if punctuation or ambiguous or unsupervised:
                        details.write(json.dumps({
                            "split": split,
                            "document_id": document["id"],
                            "language": document.get("language"),
                            "punctuation_boundary_conflicts": punctuation,
                            "ambiguous_boundary_conflicts": ambiguous,
                            "ambiguous_entities": ambiguous_entities,
                            "unsupervised_entities": unsupervised,
                        }, ensure_ascii=False) + "\n")

                    if split_index % args.progress_every == 0:
                        print(
                            f"{split}: {split_index:,}/{document_counts[split]:,}; "
                            f"punctuation={totals['punctuation_tokens']:,}; "
                            f"ambiguous={totals['ambiguous_tokens']:,}; "
                            f"errors={totals['processing_errors']:,}",
                            flush=True,
                        )

    summary = {
        "passed": totals["processing_errors"] == 0,
        "scope": {
            "data_dir": str(args.data_dir.resolve()),
            "documents": document_counts,
            "data_sha256": hashes,
            "checkpoint": args.checkpoint,
            "max_length": args.max_length,
            "stride": args.stride,
            "entity_types": entity_types,
        },
        "policy": {
            "punctuation_only_boundary": "label the whole token with BIO",
            "ambiguous_boundary": "assign -100 to every token of the affected entity",
            "chunk_boundary": "use stride; incomplete copies receive -100",
        },
        "totals": dict(totals),
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_language": {
            key: dict(value) for key, value in sorted(by_language.items())
        },
        "punctuation_tokens_by_label": dict(punctuation_by_label.most_common()),
        "ambiguous_tokens_by_label": dict(ambiguous_by_label.most_common()),
        "ambiguous_tokens_by_reason": dict(ambiguous_by_reason.most_common()),
        "punctuation_outside_text": dict(outside_text_counts.most_common()),
        "non_alphanumeric_annotations_by_label": dict(
            non_alphanumeric_annotations_by_label.most_common()
        ),
        "artifacts": {
            "cases": str(details_path.resolve()),
            "errors": str(errors_path.resolve()),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "passed": summary["passed"],
        "documents": totals["documents"],
        "punctuation_tokens": totals["punctuation_tokens"],
        "ambiguous_tokens": totals["ambiguous_tokens"],
        "ambiguous_entities": totals["ambiguous_entities"],
        "unsupervised_entities": totals["unsupervised_entities"],
        "processing_errors": totals["processing_errors"],
        "summary": str(summary_path.resolve()),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
