# Finaira bilingual NER training

`train.py` uses PyTorch DataLoaders, dynamic padding, token cross-entropy, AdamW,
and a configurable XLM-R encoder + NER head. `models/xlmr.py` supports head-only,
partial, and full fine-tuning. All three configs start with the hyperparameters reported in Section 3.4 of
*Bridging the Gap: Efficient Cross-Lingual NER in Low-Resource Financial Domain*:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Encoder learning rate | 0.00002 |
| Head learning rate | 0.00002 |
| Batch size | 28 |
| Epochs | 20 |

The paper reports one learning rate, so it is applied to both parameter groups.
These are the paper's general experimental settings, not a separately documented
teacher/freezing recipe. The configs share identical training settings; only
`model.strategy` and partial-freezing options differ.

Seed 42, weight decay 0.01, gradient clipping 1.0, maximum length 512, stride 64,
and partial unfreezing of the last four encoder layers are implementation
choices, not values established by that paper. This code uses supervised NER
cross-entropy; the paper's distillation/consistency coefficient is not applicable.

Source: https://aclanthology.org/2025.finnlp-1.5.pdf (Section 3.4).

## Configure and run

From this directory, install dependencies in your chosen virtual environment:

```sh
python -m pip install -r requirements.txt
```

Edit a config under `configs/`:

- Set `data.train_path` and `data.validation_path` to annotated JSONL files.
  Relative paths resolve against the config file's directory.
- Populate `data.entity_types` in a fixed order. All splits use that mapping.
- Set the DagsHub tracking URI and experiment name, or set
  `mlflow.enabled` to false for a local-only experiment.
- Keep credentials outside the config. For DagsHub, set
  `MLFLOW_TRACKING_USERNAME` to your username and `MLFLOW_TRACKING_PASSWORD`
  to your access token in the execution environment. `MLFLOW_TRACKING_URI`
  can override the configured endpoint.

```sh
python train.py --config configs/head_only.json
```

For Kaggle, install the dependencies in the notebook and run from this
directory:

```python
!pip install -q -r requirements.txt
!python train.py --config configs/head_only.json
```

`FacebookAI/xlm-roberta-base` is the checkpoint name, not a file that must be
committed to this repository. On the first run, Transformers downloads the
tokenizer and model weights from Hugging Face into Kaggle's local cache. Later
runs reuse that cache. If Kaggle Internet is disabled, add the checkpoint as a
Kaggle dataset or enable Internet before running.

Select a different strategy by changing only the config argument:

```sh
python train.py --config configs/partial.json
python train.py --config configs/full.json
```

The head-only configuration is set up for an initial representative trial:
it samples 5% of training documents and 10% of validation documents using a
deterministic language/entity-aware sampler, and runs for 3 epochs. The sampler
preserves proportional language coverage and greedily ensures that every entity
type present in the source split is represented when the sample is large enough.
Set `sampling.enabled` to `false` and restore the desired epoch count for a full
dataset run.

From the Finaira project root, the equivalent command is:

```sh
python "Code/Detection/NER/Bilingual Model/train.py" --config "Code/Detection/NER/Bilingual Model/configs/head_only.json"
```

Each invocation creates a unique directory under `runs/`; it starts from the
configured pretrained checkpoint. Validation runs without gradients or optimizer
updates. Frozen encoder modules use evaluation mode to disable their dropout.
The trainable head and any trainable encoder layers remain in training mode.

## Results and DagsHub upload

Each epoch saves training loss and token accuracy from the training batches,
plus validation loss, token accuracy, token-level F1, strict entity precision/recall/F1, relaxed-overlap entity
precision/recall/F1, per-entity-type scores, boundary-error counts, masked
ambiguous-entity counts, and supervised-token counts in `history.json`. The
default best-checkpoint metric is `validation_entity_strict_f1`. The model and
tokenizer are saved under `best_model/`. The run also saves its config, label
mapping, dataset hashes, aggregate coverage, and parameter counts.

After training, `upload_to_mlflow()` sends these metrics at their epoch steps and
uploads config, history, mapping, and aggregate summaries to DagsHub MLflow.
Checkpoint upload is optional (`mlflow.upload_checkpoint`). Raw datasets,
prediction text, and the detailed `coverage_local.json` are not uploaded.
No remote connection occurs until this final upload step.

If the upload fails, local results and the best model remain available. Retry:

```sh
python train.py --upload-only runs/REPLACE_WITH_RUN_DIRECTORY
```

The stored MLflow run ID allows retries to reuse the same run after it has been
created. This is experiment tracking; it does not publish/register a model.

## Evaluation meaning

These are token-level metrics over overlapping chunks, not document-level NER
precision/recall. Overlap positions can count multiple times. Unannotated text is
provisionally O even for positive-only types, so validation loss/accuracy inherit
annotation incompleteness. Best-checkpoint selection currently uses this loss.
A reviewed entity-level evaluation should be added before comparing final NER
quality. Related source documents and augmented variants must be separated by
case before training; the loader only checks shared IDs and identical text.

Preprocessing masks special tokens, ignore regions, and incomplete chunk-boundary
copies. When a token crosses an entity boundary only through Unicode punctuation
or symbols, the whole token receives the entity BIO label. When the outside part
contains letters/numbers or one token overlaps multiple entities, the complete
affected entity receives `-100`; this avoids training on a truncated gold span.
All cases are included in the local coverage report.

## Full tokenizer alignment audit

Run the production XLM-R tokenizer and alignment policy over all three mixed-data
splits:

```sh
python tokenizerAudit.py
```

The command writes aggregate counts to
`tokenizer_audit_results/tokenizer_audit_summary.json`, individual affected
documents to `tokenizer_alignment_cases.jsonl`, and unexpected exceptions to
`tokenizer_processing_errors.jsonl`. Stride-overlap copies are deduplicated by
their original document offsets.

`span_postprocessing.py` contains the explicit list of entity types whose outer
punctuation may be trimmed after BIO decoding. It never removes punctuation from
inside a decoded span and intentionally leaves punctuation-sensitive types such
as phone numbers unchanged.

## Verification

```sh
python -m unittest discover -s tests -v
```

The tests use a tiny PyTorch network and synthetic batches; they do not download
XLM-R, train on the real dataset, or contact DagsHub.

Official tracking setup: https://dagshub.com/docs/integration_guide/mlflow_tracking/
