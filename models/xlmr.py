"""Construct XLM-R with an NER head and configure its freezing strategy."""

from transformers import AutoModelForTokenClassification


def build_model(config, label2id):
    if config.get("architecture", "xlmr") != "xlmr":
        raise ValueError("Only the xlmr architecture is currently implemented.")
    model = AutoModelForTokenClassification.from_pretrained(
        config["checkpoint"],
        num_labels=len(label2id),
        label2id=label2id,
        id2label={index: label for label, index in label2id.items()},
        cache_dir=config.get("cache_dir"),
    )
    strategy = config["strategy"]
    if strategy not in {"head_only", "partial", "full"}:
        raise ValueError("strategy must be head_only, partial, or full.")

    # The new classification head remains trainable in every strategy.
    for parameter in model.roberta.parameters():
        parameter.requires_grad = strategy == "full"

    if strategy == "partial":
        layers = model.roberta.encoder.layer
        count = config["train_last_n_layers"]
        if not isinstance(count, int) or not 1 <= count <= len(layers):
            raise ValueError("train_last_n_layers is outside the encoder range.")
        for layer in layers[-count:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        for parameter in model.roberta.embeddings.parameters():
            parameter.requires_grad = config.get("train_embeddings", False)
    return model
