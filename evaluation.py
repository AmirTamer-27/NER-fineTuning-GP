"""Token and entity-level evaluation for the bilingual XLM-R NER model."""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from span_postprocessing import trim_decoded_span


MODEL_FIELDS = ("input_ids", "attention_mask", "labels")


def model_only_examples(examples):
    """Remove diagnostic metadata before examples reach the data collator."""
    return [{field: example[field] for field in MODEL_FIELDS} for example in examples]


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _prf(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def token_metrics(confusion, id2label):
    total = sum(confusion.values())
    correct = sum(value for (gold, predicted), value in confusion.items() if gold == predicted)
    entity_labels = [label_id for label_id, label in id2label.items() if label != "O"]
    per_label = {}
    for label_id in entity_labels:
        tp = confusion[(label_id, label_id)]
        fp = sum(value for (gold, predicted), value in confusion.items() if predicted == label_id and gold != label_id)
        fn = sum(value for (gold, predicted), value in confusion.items() if gold == label_id and predicted != label_id)
        per_label[id2label[label_id]] = _prf(tp, fp, fn)
    macro = {
        metric: sum(values[metric] for values in per_label.values()) / len(per_label)
        if per_label else 0.0
        for metric in ("precision", "recall", "f1")
    }
    tp = sum(confusion[(label_id, label_id)] for label_id in entity_labels)
    fp = sum(
        value for (gold, predicted), value in confusion.items()
        if predicted in entity_labels and gold != predicted
    )
    fn = sum(
        value for (gold, predicted), value in confusion.items()
        if gold in entity_labels and gold != predicted
    )
    return {
        "accuracy": correct / total if total else 0.0,
        "supervised_tokens": total,
        "micro": _prf(tp, fp, fn),
        "macro": macro,
        "per_label": per_label,
    }


def token_label_counts(confusion, id2label):
    """Count gold/predicted tags only where the gold token is supervised."""
    return {
        id2label[label_id]: {
            "gold": sum(count for (gold, _), count in confusion.items() if gold == label_id),
            "predicted": sum(count for (_, predicted), count in confusion.items() if predicted == label_id),
        }
        for label_id in sorted(id2label)
    }


def invalid_i_transitions(predicted_ids, gold_ids, id2label):
    """Count I-tags that don't follow a B/I-tag of the same type in a chunk."""
    invalid = 0
    previous = "O"
    for predicted_id, gold_id in zip(predicted_ids, gold_ids):
        if gold_id == -100:
            previous = "O"
            continue
        tag = id2label[int(predicted_id)]
        if tag.startswith("I-") and previous not in ("B-" + tag[2:], tag):
            invalid += 1
        previous = tag
    return invalid


def _decode_chunk(predicted_ids, example, id2label, text):
    spans = []
    current = None

    def close_current():
        nonlocal current
        if current is None:
            return
        start, end, label = current
        start, end = trim_decoded_span(text, start, end, label)
        if start < end:
            spans.append((start, end, label))
        current = None

    for predicted_id, gold_id, (start, end) in zip(
        predicted_ids, example["labels"], example["offset_mapping"]
    ):
        # This excludes special tokens, ignored regions, incomplete chunks and
        # ambiguous entities. The model never predicts -100 itself.
        if gold_id == -100 or start == end:
            close_current()
            continue
        tag = id2label[int(predicted_id)]
        if tag == "O":
            close_current()
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or current is None or current[2] != label:
            close_current()
            current = [start, end, label]
        else:
            current[1] = max(current[1], end)
    close_current()
    return spans


def _ignored_and_ambiguous(record, coverage):
    ignored_regions = record.get("ignore_spans", [])
    ambiguous = {
        (span["start"], span["end"], span["label"])
        for span in coverage.get("ambiguous_entities", [])
    }
    ignored = set()
    for span in record.get("spans", []):
        if any(
            span["start"] < region["end"] and region["start"] < span["end"]
            for region in ignored_regions
        ):
            ignored.add((span["start"], span["end"], span["label"]))
    return ignored, ambiguous, ignored_regions, [span for span in coverage.get("ambiguous_entities", [])]


def _relaxed_matches(gold, predicted):
    """Greedy one-to-one same-label overlap matching for diagnostic F1."""
    remaining = set(gold)
    matched = set()
    for prediction in sorted(predicted):
        candidates = [
            target for target in remaining
            if target[2] == prediction[2]
            and target[0] < prediction[1]
            and prediction[0] < target[1]
        ]
        if candidates:
            target = max(candidates, key=lambda item: min(item[1], prediction[1]) - max(item[0], prediction[0]))
            remaining.remove(target)
            matched.add(prediction)
    return matched


def entity_metrics(records, reports, predictions_by_document):
    exact_gold = set()
    predicted = set()
    masked_entities = 0
    ignored_entities = 0
    boundary_errors = 0
    per_label_counts = defaultdict(lambda: Counter())

    for record, coverage in zip(records, reports):
        ignored, ambiguous, ignored_regions, ambiguous_records = _ignored_and_ambiguous(record, coverage)
        masked_entities += len(ambiguous)
        ignored_entities += len(ignored)
        document_id = record["id"]
        for span in record.get("spans", []):
            key = (span["start"], span["end"], span["label"])
            if key not in ignored and key not in ambiguous:
                exact_gold.add((document_id, *key))
        for start, end, label in predictions_by_document.get(document_id, set()):
            if any(start < region["end"] and region["start"] < end for region in ignored_regions):
                continue
            if any(start < span["end"] and span["start"] < end for span in ambiguous_records):
                continue
            predicted.add((document_id, start, end, label))

    true_positive = exact_gold & predicted
    strict = _prf(len(true_positive), len(predicted - exact_gold), len(exact_gold - predicted))
    relaxed_gold = {(doc, start, end, label) for doc, start, end, label in exact_gold}
    relaxed_predicted = {(doc, start, end, label) for doc, start, end, label in predicted}
    relaxed_matches = set()
    for document_id in {key[0] for key in relaxed_gold | relaxed_predicted}:
        gold_local = {(s, e, label) for doc, s, e, label in relaxed_gold if doc == document_id}
        pred_local = {(s, e, label) for doc, s, e, label in relaxed_predicted if doc == document_id}
        relaxed_matches.update((document_id, *key) for key in _relaxed_matches(gold_local, pred_local))
    relaxed = _prf(
        len(relaxed_matches),
        len(predicted) - len(relaxed_matches),
        len(exact_gold) - len(relaxed_matches),
    )

    for prediction in predicted - true_positive:
        document_id, start, end, label = prediction
        if any(
            document_id == gold_doc and label == gold_label
            and start < gold_end and gold_start < end
            for gold_doc, gold_start, gold_end, gold_label in exact_gold
        ):
            boundary_errors += 1

    for key in sorted({key[3] for key in exact_gold | predicted}):
        gold_label = {item for item in exact_gold if item[3] == key}
        predicted_label = {item for item in predicted if item[3] == key}
        matched = gold_label & predicted_label
        per_label_counts[key] = _prf(
            len(matched), len(predicted_label - gold_label), len(gold_label - predicted_label)
        )

    return {
        "strict": strict,
        "relaxed_overlap": relaxed,
        "per_label": dict(per_label_counts),
        "gold_entities": len(exact_gold),
        "predicted_entities": len(predicted),
        "masked_ambiguous_entities": masked_entities,
        "ignored_entities": ignored_entities,
        "boundary_errors_with_correct_type": boundary_errors,
    }


def entity_metrics_by_language(records, reports, predictions_by_document):
    """Score spans in Arabic and English documents using the same entity rules."""
    by_language = {}
    for language in ("ar", "en"):
        pairs = [
            (record, report) for record, report in zip(records, reports)
            if record.get("language") == language
        ]
        if pairs:
            by_language[language] = entity_metrics(
                [record for record, _ in pairs],
                [report for _, report in pairs],
                predictions_by_document,
            )
    return by_language


@torch.no_grad()
def evaluate_model(
    model, examples, records, reports, collator, id2label, batch_size, device,
    progress_desc="Evaluating",
):
    model.eval()
    model_examples = model_only_examples(examples)
    loader = DataLoader(model_examples, batch_size=batch_size, shuffle=False, collate_fn=collator)
    confusion = Counter()
    predictions_by_document = defaultdict(set)
    total_loss = 0.0
    total_tokens = 0
    invalid_i_count = 0
    row_predictions = []
    for batch in tqdm(loader, desc=progress_desc, unit="batch"):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"]
        supervised = labels != -100
        count = supervised.sum().item()
        total_loss += outputs.loss.item() * count
        total_tokens += count
        predicted_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
        row_predictions.extend(predicted_ids)
        label_rows = labels.cpu().tolist()
        for predicted_row, gold_row in zip(predicted_ids, label_rows):
            invalid_i_count += invalid_i_transitions(predicted_row, gold_row, id2label)
            for predicted, gold in zip(predicted_row, gold_row):
                if gold != -100:
                    confusion[(int(gold), int(predicted))] += 1

    # Decode using the original document text. Raw text is kept outside model
    # tensors and is used only for character offsets and punctuation cleanup.
    text_by_id = {record["id"]: record["text"] for record in records}
    for example, predicted in zip(examples, row_predictions):
        for span in _decode_chunk(predicted, example, id2label, text_by_id[example["document_id"]]):
            predictions_by_document[example["document_id"]].add(span)

    token = token_metrics(confusion, id2label)
    entities = entity_metrics(records, reports, predictions_by_document)
    entities_by_language = entity_metrics_by_language(records, reports, predictions_by_document)
    return {
        "loss": total_loss / total_tokens if total_tokens else 0.0,
        "token_accuracy": token["accuracy"],
        "supervised_tokens": total_tokens,
        "token_micro_f1": token["micro"]["f1"],
        "token_macro_f1": token["macro"]["f1"],
        "token_per_label": token["per_label"],
        "token_label_counts": token_label_counts(confusion, id2label),
        "invalid_i_transitions": invalid_i_count,
        "entity": entities,
        "entity_by_language": entities_by_language,
    }
