"""Deterministic language/entity-aware document sampling for quick experiments."""

from collections import defaultdict
import random


def stratified_sample(records, fraction, seed=42):
    """Sample documents while preserving language and entity-type coverage.

    The sample size is based on documents, before tokenization. Language quotas
    are proportional to the source split, and a greedy first pass ensures that
    every language and every entity type present in the source is represented
    whenever the requested sample is large enough.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Sampling fraction must be in (0, 1].")
    if fraction == 1:
        return list(records)

    rng = random.Random(seed)
    total = len(records)
    sample_size = max(1, round(total * fraction))
    languages = defaultdict(list)
    entity_to_indices = defaultdict(list)
    for index, record in enumerate(records):
        language = record.get("language", "unknown")
        languages[language].append(index)
        for span in record.get("spans", []):
            entity_to_indices[span["label"]].append(index)

    # Proportional language quotas using largest remainder allocation.
    quotas = {
        language: max(1, sample_size * len(indices) // total)
        for language, indices in languages.items()
    }
    while sum(quotas.values()) > sample_size:
        language = max(quotas, key=lambda item: quotas[item])
        if quotas[language] > 1:
            quotas[language] -= 1
        else:
            break
    while sum(quotas.values()) < sample_size:
        language = max(languages, key=lambda item: len(languages[item]) - quotas[item])
        quotas[language] += 1

    shuffled_by_language = {}
    for language, indices in languages.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        shuffled_by_language[language] = shuffled

    selected = set()
    selected_by_language = defaultdict(int)

    def add(index):
        language = records[index].get("language", "unknown")
        if index in selected or selected_by_language[language] >= quotas[language]:
            return False
        selected.add(index)
        selected_by_language[language] += 1
        return True

    # First guarantee one document per language and then cover rare labels.
    for language, indices in shuffled_by_language.items():
        add(indices[0])
    missing_labels = set(entity_to_indices)
    while missing_labels and len(selected) < sample_size:
        candidates = [
            index for label in missing_labels for index in entity_to_indices[label]
            if index not in selected
        ]
        candidates = list(set(candidates))
        rng.shuffle(candidates)
        candidate = max(
            candidates,
            key=lambda index: sum(
                1 for span in records[index].get("spans", [])
                if span["label"] in missing_labels
            ),
            default=None,
        )
        if candidate is None or not add(candidate):
            break
        covered = {span["label"] for span in records[candidate].get("spans", [])}
        missing_labels -= covered

    # Fill the remaining quota with the deterministic language-shuffled pools.
    for language, indices in shuffled_by_language.items():
        for index in indices:
            if len(selected) >= sample_size:
                break
            add(index)

    return [records[index] for index in sorted(selected)]
