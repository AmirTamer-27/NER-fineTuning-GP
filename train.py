"""Config-driven PyTorch NER training with a final DagsHub MLflow upload."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation import evaluate_model, model_only_examples
from prepare_data import prepare_ner_dataset
from sampling import stratified_sample


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_records(path):
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_optimizer(model, config):
    if config.get("optimizer", "AdamW").lower() != "adamw":
        raise ValueError("Only the AdamW optimizer is currently supported.")
    encoder_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = encoder_parameters if name.startswith("roberta.") else head_parameters
            group.append(parameter)
    groups = []
    if encoder_parameters:
        groups.append({"params": encoder_parameters, "lr": config["encoder_learning_rate"]})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": config["head_learning_rate"]})
    if not groups:
        raise ValueError("No trainable parameters were selected.")
    return AdamW(groups, weight_decay=config.get("weight_decay", 0.01))


def run_epoch(
    model, dataloader, device, optimizer=None, max_grad_norm=1.0,
    progress_desc="Training",
):
    training = optimizer is not None
    model.train(training)
    # Frozen features stay deterministic: training mode alone enables dropout.
    if training and hasattr(model, "roberta"):
        if not any(p.requires_grad for p in model.roberta.embeddings.parameters()):
            model.roberta.embeddings.eval()
        for layer in model.roberta.encoder.layer:
            if not any(p.requires_grad for p in layer.parameters()):
                layer.eval()

    total_loss = 0.0
    total_labelled_tokens = 0
    total_correct = 0
    for batch in tqdm(dataloader, desc=progress_desc, unit="batch"):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        supervised = batch["labels"] != -100
        labelled_tokens = supervised.sum().item()
        if labelled_tokens == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(**batch)
            loss = outputs.loss  # Token cross-entropy; -100 positions ignored.
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")
            if training:
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()

        total_loss += loss.item() * labelled_tokens
        total_labelled_tokens += labelled_tokens
        predictions = outputs.logits.detach().argmax(dim=-1)
        total_correct += ((predictions == batch["labels"]) & supervised).sum().item()

    if total_labelled_tokens == 0:
        raise ValueError("No supervised tokens were found in this epoch.")
    return {
        "loss": total_loss / total_labelled_tokens,
        "token_accuracy": total_correct / total_labelled_tokens,
        "supervised_tokens": total_labelled_tokens,
    }


def coverage_summary(report):
    return {
        "documents": len(report),
        "target_entities": sum(r["target_entities"] for r in report),
        "supervised_entities": sum(r["supervised_entities"] for r in report),
        "ignored_entities": sum(r["ignored_entities"] for r in report),
        "punctuation_boundary_tokens": sum(
            len(r["punctuation_boundary_conflicts"]) for r in report
        ),
        "ambiguous_boundary_tokens": sum(
            len({
                (item["token_start"], item["token_end"])
                for item in r["ambiguous_boundary_conflicts"]
            })
            for r in report
        ),
        "ambiguous_entities": sum(len(r["ambiguous_entities"]) for r in report),
        "uncovered_entities": sum(len(r["unsupervised_entities"]) for r in report),
    }


def flatten_evaluation(prefix, result):
    """Convert nested evaluation output into MLflow-friendly scalar metrics."""
    flattened = {
        f"{prefix}_loss": result["loss"],
        f"{prefix}_token_accuracy": result["token_accuracy"],
        f"{prefix}_token_micro_f1": result["token_micro_f1"],
        f"{prefix}_token_macro_f1": result["token_macro_f1"],
        f"{prefix}_invalid_i_transitions": result["invalid_i_transitions"],
        f"{prefix}_entity_strict_precision": result["entity"]["strict"]["precision"],
        f"{prefix}_entity_strict_recall": result["entity"]["strict"]["recall"],
        f"{prefix}_entity_strict_f1": result["entity"]["strict"]["f1"],
        f"{prefix}_entity_relaxed_precision": result["entity"]["relaxed_overlap"]["precision"],
        f"{prefix}_entity_relaxed_recall": result["entity"]["relaxed_overlap"]["recall"],
        f"{prefix}_entity_relaxed_f1": result["entity"]["relaxed_overlap"]["f1"],
        f"{prefix}_evaluable_gold_entities": result["entity"]["gold_entities"],
        f"{prefix}_predicted_entities": result["entity"]["predicted_entities"],
        f"{prefix}_masked_ambiguous_entities": result["entity"]["masked_ambiguous_entities"],
        f"{prefix}_ignored_entities": result["entity"]["ignored_entities"],
        f"{prefix}_boundary_errors": result["entity"]["boundary_errors_with_correct_type"],
    }
    for label, metrics in result["entity"]["per_label"].items():
        safe_label = label.lower()
        flattened[f"{prefix}_{safe_label}_precision"] = metrics["precision"]
        flattened[f"{prefix}_{safe_label}_recall"] = metrics["recall"]
        flattened[f"{prefix}_{safe_label}_f1"] = metrics["f1"]
    for label, counts in result["token_label_counts"].items():
        safe_label = label.lower().replace("-", "_")
        flattened[f"{prefix}_token_{safe_label}_gold"] = counts["gold"]
        flattened[f"{prefix}_token_{safe_label}_predicted"] = counts["predicted"]
    return flattened


def train(config_path):
    # Lazy imports allow --help and upload-only without downloading a model.
    from transformers import AutoTokenizer, DataCollatorForTokenClassification, set_seed
    from models import build_model

    config = json.loads(config_path.read_text())
    settings = config["training"]
    if settings["epochs"] < 1 or settings["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if not config["data"]["entity_types"]:
        raise ValueError("Populate data.entity_types with the selected target types.")
    for split in ("train", "validation"):
        if not config["data"].get(f"{split}_path"):
            raise ValueError(f"Populate data.{split}_path in the config.")

    random.seed(settings["seed"])
    set_seed(settings["seed"])
    base = config_path.parent
    paths = {s: (base / config["data"][f"{s}_path"]).resolve() for s in ("train", "validation")}
    records = {s: load_records(p) for s, p in paths.items()}
    train_ids = {r["id"] for r in records["train"]}
    if train_ids.intersection(r["id"] for r in records["validation"]):
        raise ValueError("Training and validation share document IDs.")
    if {r["text"] for r in records["train"]}.intersection(r["text"] for r in records["validation"]):
        raise ValueError("Training and validation contain identical document text.")

    sampling = config.get("sampling", {})
    if sampling.get("enabled", False):
        sample_seed = sampling.get("seed", settings["seed"])
        records["train"] = stratified_sample(
            records["train"], sampling.get("train_fraction", 1.0), sample_seed
        )
        records["validation"] = stratified_sample(
            records["validation"], sampling.get("validation_fraction", 1.0), sample_seed + 1
        )
        print(
            "Using stratified document sample: "
            f"train={len(records['train']):,}, "
            f"validation={len(records['validation']):,}"
        )

    cache_dir = config["model"].get("cache_dir")
    if cache_dir:
        cache_dir = str((base / cache_dir).resolve())
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["checkpoint"], use_fast=True, cache_dir=cache_dir
    )
    prepared_examples, examples, reports = {}, {}, {}
    label2id = None
    for split in ("train", "validation"):
        prepared_examples[split], mapping, reports[split] = prepare_ner_dataset(
            records[split], tokenizer, config["data"]["entity_types"],
            max_length=config["data"]["max_length"],
            stride=config["data"]["stride"],
            include_metadata=True,
            progress_desc=f"Tokenizing {split}",
        )
        if not prepared_examples[split]:
            raise ValueError(f"No prepared examples for {split}.")
        examples[split] = model_only_examples(prepared_examples[split])
        if label2id is not None and mapping != label2id:
            raise ValueError("Training and validation label mappings differ.")
        label2id = mapping

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
        return_tensors="pt",
    )
    train_loader = DataLoader(
        examples["train"], batch_size=settings["batch_size"], shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(settings["seed"]),
    )
    device = resolve_device(settings.get("device", "auto"))
    model = build_model(config["model"], label2id).to(device)
    optimizer = make_optimizer(model, settings)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_dir = (base / settings["output_dir"] / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "label2id.json", label2id)
    summary = {
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "data": {
            s: {**coverage_summary(reports[s]), "chunks": len(examples[s]), "sha256": file_digest(paths[s])}
            for s in paths
        },
        "validation_note": "Entity metrics decode document spans, deduplicate stride overlaps, trim configured outer punctuation, and exclude ambiguous/ignored gold regions.",
    }
    write_json(run_dir / "summary.json", summary)
    # Full coverage reports may contain source spans: retain them only locally.
    write_json(run_dir / "coverage_local.json", reports)
    print(f"Run directory: {run_dir}")
    history = []
    evaluation_history = []
    selection_metric = config.get("evaluation", {}).get(
        "selection_metric", "validation_loss"
    )
    maximize_selection = any(
        term in selection_metric for term in ("f1", "precision", "recall", "accuracy")
    )
    best_score = -math.inf if maximize_selection else math.inf
    for epoch in range(1, settings["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer,
            settings.get("max_grad_norm", 1.0),
            progress_desc=f"Epoch {epoch}/{settings['epochs']} - train",
        )
        validation_evaluation = evaluate_model(
            model, prepared_examples["validation"], records["validation"], reports["validation"],
            collator, {value: key for key, value in label2id.items()},
            config.get("evaluation", {}).get("batch_size", settings["batch_size"]), device,
            progress_desc=f"Epoch {epoch}/{settings['epochs']} - validation",
        )
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update(flatten_evaluation("validation", validation_evaluation))
        history.append(row)
        write_json(run_dir / "history.json", history)
        evaluation_row = {
            "epoch": epoch,
            "validation": validation_evaluation,
        }
        evaluation_history.append(evaluation_row)
        write_json(run_dir / "evaluation_history.json", evaluation_history)
        write_json(run_dir / f"evaluation_epoch_{epoch}.json", evaluation_row)
        if selection_metric not in row:
            raise ValueError(f"Unknown evaluation selection metric: {selection_metric}")
        score = row[selection_metric]
        is_better = score > best_score if maximize_selection else score < best_score
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_accuracy={train_metrics['token_accuracy']:.2%}, "
            f"validation_loss={validation_evaluation['loss']:.4f}, "
            f"validation_accuracy={validation_evaluation['token_accuracy']:.2%}, "
            f"validation_recall={validation_evaluation['entity']['strict']['recall']:.2%}"
        )
        if is_better:
            best_score = score
            model.save_pretrained(run_dir / "best_model")
            tokenizer.save_pretrained(run_dir / "best_model")
            write_json(run_dir / "best_metrics.json", row)
            write_json(run_dir / "best_evaluation.json", evaluation_row)

    upload_to_mlflow(run_dir)
    return run_dir


# Final step: upload training/validation results to DagsHub's MLflow experiment.
def upload_to_mlflow(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    tracking = config["mlflow"]
    if not tracking.get("enabled", False):
        print("MLflow upload disabled; results are saved locally.")
        return
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI") or tracking.get("tracking_uri")
    if not uri or "YOUR_" in uri:
        raise ValueError("Set mlflow.tracking_uri or MLFLOW_TRACKING_URI to your DagsHub endpoint.")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(tracking["experiment_name"])
    history = json.loads((run_dir / "history.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    best = json.loads((run_dir / "best_metrics.json").read_text())
    run_id_path = run_dir / "mlflow_run_id.txt"
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else None
    with mlflow.start_run(run_id=run_id, run_name=run_dir.name) as run:
        run_id_path.write_text(run.info.run_id + "\n")
        params = {f"model.{k}": v for k, v in config["model"].items()}
        params.update({f"training.{k}": v for k, v in config["training"].items()})
        params.update({f"evaluation.{k}": v for k, v in config.get("evaluation", {}).items()})
        params.update({"max_length": config["data"]["max_length"], "stride": config["data"]["stride"],
                       "parameters": summary["parameters"], "trainable_parameters": summary["trainable_parameters"]})
        for split, info in summary["data"].items():
            params.update({f"{split}.{k}": v for k, v in info.items()})
        mlflow.log_params(params)
        for row in history:
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=row["epoch"])
        mlflow.log_metrics({"best_epoch": best["epoch"], "best_validation_loss": best["validation_loss"]})
        for name in (
            "config.json", "label2id.json", "summary.json", "history.json",
            "evaluation_history.json", "best_metrics.json", "best_evaluation.json",
        ):
            artifact = run_dir / name
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="results")
        if tracking.get("upload_checkpoint", False):
            mlflow.log_artifacts(str(run_dir / "best_model"), artifact_path="best_model")
        print(f"Uploaded results to MLflow run {run.info.run_id}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Example: python train.py --config configs/head_only.json",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path, help="Training JSON config (head_only, partial, or full).")
    group.add_argument("--upload-only", type=Path, help="Retry upload of a saved run without retraining.")
    args = parser.parse_args()
    if args.upload_only:
        upload_to_mlflow(args.upload_only.resolve())
    else:
        train(args.config.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
