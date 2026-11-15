"""Back up completed best-model checkpoints inside the active MLflow run."""

import json


def upload_best_checkpoint(run_dir, tracking, best_metrics):
    """Publish a versioned checkpoint, then publish its completion pointer.

    Earlier epoch directories remain intact if a later upload is interrupted.
    Only successful uploads update the local receipt and remote pointer.
    """
    epoch = int(best_metrics['epoch'])
    receipt_path = run_dir / 'checkpoint_upload.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['epoch'] == epoch:
            return receipt
    model_dir = run_dir / 'best_model'
    has_weights = any(model_dir.glob('*.safetensors')) or any(model_dir.glob('pytorch_model*.bin'))
    if not (model_dir / 'config.json').is_file() or not has_weights:
        raise FileNotFoundError(f'No complete model configuration and weights found in {model_dir}')
    artifact_dir = f'checkpoints/epoch_{epoch:04d}'
    model_artifact = f'{artifact_dir}/best_model'
    print(f'Uploading best checkpoint from epoch {epoch} to MLflow...', flush=True)
    tracking.log_artifacts(str(model_dir), artifact_path=model_artifact)
    # Store run settings and the selected epoch alongside the model.
    tracking.log_artifact(str(run_dir / 'config.json'), artifact_path=artifact_dir)
    tracking.log_artifact(str(run_dir / 'best_metrics.json'), artifact_path=artifact_dir)
    receipt = {'epoch': epoch, 'artifact_path': model_artifact}
    # A model is advertised as downloadable only after all its uploads succeed.
    tracking.log_dict(receipt, 'results/latest_checkpoint.json')
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'Checkpoint backed up: {model_artifact}', flush=True)
    return receipt
