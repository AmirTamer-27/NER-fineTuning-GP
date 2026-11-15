"""Prepare unpadded BIO training examples from annotated document records."""

from alignment_policy import classify_boundary_conflict
from tokenization import tokenize_document, validate_tokenization_settings
from tqdm.auto import tqdm


def prepare_ner_dataset(
    dataset,
    tokenizer,
    entity_types,
    max_length=512,
    stride=64,
    include_metadata=False,
    progress_desc="Tokenizing",
):
    """Return (examples, label2id, coverage_report).

    All subwords of annotated entities are supervised. Unannotated text is
    provisionally O, including types with positive-only annotations. Original
    documents and completeness metadata are not modified. Overlap positions
    can receive supervision in multiple chunks; no deduplication is applied.
    """
    validate_tokenization_settings(tokenizer, max_length, stride)

    entity_types = list(entity_types)
    if not entity_types or len(entity_types) != len(set(entity_types)):
        raise ValueError("Supply a nonempty, unique list of target entity types.")
    tags = ["O"]
    for entity_type in entity_types:
        tags.extend([f"B-{entity_type}", f"I-{entity_type}"])
    label2id = {tag: index for index, tag in enumerate(tags)}
    target_types = set(entity_types)
    prepared = []
    coverage_report = []

    def overlaps(start, end, span):
        return start < span["end"] and span["start"] < end

    for document in tqdm(dataset, desc=progress_desc, unit="doc"):
        document_id = document["id"]
        text = document["text"]
        ignored = document.get("ignore_spans", [])
        spans = sorted(
            [s for s in document["spans"] if s["label"] in target_types],
            key=lambda s: s["start"],
        )
        for span in [*spans, *ignored]:
            start, end = span["start"], span["end"]
            if not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid span in {document_id}: {span}")
            if "text" in span and text[start:end] != span["text"]:
                raise ValueError(f"Span text mismatch in {document_id}: {span}")
        for previous, current in zip(spans, spans[1:]):
            if current["start"] < previous["end"]:
                raise ValueError(f"Overlapping target entities in {document_id}")

        ignored_entities = {
            index for index, span in enumerate(spans)
            if any(overlaps(span["start"], span["end"], s) for s in ignored)
        }
        supervised_entities = set()
        chunks = tokenize_document(
            text,
            tokenizer,
            max_length=max_length,
            stride=stride,
        )

        punctuation_conflicts = {}
        ambiguous_conflicts = {}
        ambiguous_entities = set()

        # Classify boundary-crossing tokens once at document level. Overflow
        # chunks repeat stride tokens, so dictionaries deduplicate by the
        # original document offsets and entity index.
        for chunk_index, offsets in enumerate(chunks["offset_mapping"]):
            special_mask = chunks["special_tokens_mask"][chunk_index]
            for (start, end), special in zip(offsets, special_mask):
                if special or start == end:
                    continue
                if any(overlaps(start, end, region) for region in ignored):
                    continue
                matches = [
                    (index, span)
                    for index, span in enumerate(spans)
                    if overlaps(start, end, span)
                ]
                if not matches:
                    continue
                if len(matches) > 1:
                    for entity_index, span in matches:
                        key = (entity_index, start, end)
                        ambiguous_entities.add(entity_index)
                        ambiguous_conflicts[key] = {
                            "chunk_index": chunk_index,
                            "token_start": start,
                            "token_end": end,
                            "token_text": text[start:end],
                            "outside_text": "",
                            "reason": "token_overlaps_multiple_entities",
                            "entity": span.copy(),
                        }
                    continue

                entity_index, span = matches[0]
                kind, outside_text = classify_boundary_conflict(
                    text,
                    start,
                    end,
                    span["start"],
                    span["end"],
                )
                if kind == "exact":
                    continue
                key = (entity_index, start, end)
                record = {
                    "chunk_index": chunk_index,
                    "token_start": start,
                    "token_end": end,
                    "token_text": text[start:end],
                    "outside_text": outside_text,
                    "reason": f"{kind}_boundary_crossing",
                    "entity": span.copy(),
                }
                if kind == "punctuation":
                    punctuation_conflicts[key] = record
                else:
                    ambiguous_entities.add(entity_index)
                    ambiguous_conflicts[key] = record

        # If any token boundary is ambiguous, ignore the whole entity. This
        # prevents training on a truncated suffix or prefix of the gold span.
        effective_punctuation_conflicts = {
            key: value
            for key, value in punctuation_conflicts.items()
            if key[0] not in ambiguous_entities
        }

        for chunk_index, offsets in enumerate(chunks["offset_mapping"]):
            special_mask = chunks["special_tokens_mask"][chunk_index]
            real_offsets = [
                (start, end)
                for (start, end), special in zip(offsets, special_mask)
                if not special and start < end
            ]
            if not real_offsets:
                continue
            chunk_start = min(start for start, end in real_offsets)
            chunk_end = max(end for start, end in real_offsets)
            labels = []
            seen_entities = set()

            for (start, end), special in zip(offsets, special_mask):
                if special or start == end:
                    labels.append(-100)
                    continue
                if any(overlaps(start, end, s) for s in ignored):
                    labels.append(-100)
                    continue
                match = next(
                    ((i, s) for i, s in enumerate(spans) if overlaps(start, end, s)),
                    None,
                )
                if match is None:
                    labels.append(label2id["O"])
                    continue
                entity_index, span = match
                if (
                    entity_index in ignored_entities
                    or entity_index in ambiguous_entities
                    or span["start"] < chunk_start
                    or span["end"] > chunk_end
                ):
                    labels.append(-100)
                    continue
                boundary_kind, _ = classify_boundary_conflict(
                    text, start, end, span["start"], span["end"]
                )
                if boundary_kind == "ambiguous":
                    labels.append(-100)
                    continue
                prefix = "I" if entity_index in seen_entities else "B"
                labels.append(label2id[f"{prefix}-{span['label']}"])
                seen_entities.add(entity_index)

            supervised_entities.update(seen_entities)
            if any(label != -100 for label in labels):
                example = {
                    "input_ids": chunks["input_ids"][chunk_index],
                    "attention_mask": chunks["attention_mask"][chunk_index],
                    "labels": labels,
                }
                if include_metadata:
                    # Diagnostic metadata is opt-in so the normal training
                    # collator continues to receive tensors only.
                    example.update({
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "offset_mapping": [list(offset) for offset in offsets],
                        "special_tokens_mask": list(special_mask),
                    })
                prepared.append(example)

        missing = (
            set(range(len(spans)))
            - ignored_entities
            - ambiguous_entities
            - supervised_entities
        )
        coverage_report.append({
            "document_id": document_id,
            "target_entities": len(spans),
            "supervised_entities": len(supervised_entities),
            "ignored_entities": len(ignored_entities),
            "ambiguous_entities": [
                spans[index].copy() for index in sorted(ambiguous_entities)
            ],
            "punctuation_boundary_conflicts": list(
                effective_punctuation_conflicts.values()
            ),
            "ambiguous_boundary_conflicts": list(ambiguous_conflicts.values()),
            "unsupervised_entities": [spans[i].copy() for i in sorted(missing)],
        })
    return prepared, label2id, coverage_report
