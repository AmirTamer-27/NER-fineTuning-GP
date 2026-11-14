"""Load XLM-R Base and prepare a placeholder annotated dataset."""

from transformers import AutoModel, AutoTokenizer

from prepare_data import prepare_ner_dataset


MODEL_NAME = "FacebookAI/xlm-roberta-base"

# Replace with annotated records containing id, text, spans, and ignore_spans.
dataset: list[dict] = []
# Supply the selected target types in a fixed order for all dataset splits.
entity_types: list[str] = []


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    print("XLM-R Base and tokenizer loaded successfully.")

    if dataset:
        prepared_dataset, label2id, coverage_report = prepare_ner_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            entity_types=entity_types,
            max_length=512,
            stride=64,
        )
        print("Prepared chunks:", len(prepared_dataset))
        print("Label mapping:", label2id)
        print("Coverage report:", coverage_report)
    else:
        print("Populate dataset and entity_types to run preprocessing.")
