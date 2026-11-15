"""Shared unpadded tokenization for training preparation and inference."""


def validate_tokenization_settings(tokenizer, max_length, stride):
    """Validate chunk capacity, including the tokenizer's special tokens."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for character offsets.")
    if not 2 < max_length <= tokenizer.model_max_length:
        raise ValueError("Invalid maximum sequence length.")
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= stride < capacity:
        raise ValueError("stride must be smaller than the text-token capacity.")


def tokenize_document(text, tokenizer, *, max_length, stride):
    """Return overflow chunks, IDs, attention masks, offsets and special masks.

    Offsets refer to the original document. No annotation alignment, BIO labels,
    padding or model tensors are created here; callers handle those separately.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    validate_tokenization_settings(tokenizer, max_length, stride)
    return tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        padding=False,
    )
