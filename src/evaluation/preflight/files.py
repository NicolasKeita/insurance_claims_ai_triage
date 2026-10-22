"""Inspect local prerequisites without importing inference frameworks or weights."""

import json
import os
from pathlib import Path

from evaluation.preflight.config import ROOT


def require_file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Required file missing or empty: {path}")
    return path


def scenario_files(scenario):
    """Synthetic PDFs are generated later; their source claim must already exist."""
    for filename in scenario.files:
        require_file(ROOT / filename)
    source = ROOT / scenario.claim_source
    source = source / "claim.json" if source.is_dir() else source
    payload = json.loads(require_file(source).read_text(encoding="utf-8"))
    if scenario.inputs.get("use_reference_case"):
        for document in payload.get("documents", []):
            if document.get("available", True):
                require_file(source.parent / "documents" / document["filename"])
    for asset in scenario.external_assets:
        if asset.get("path"):
            require_file(ROOT / asset["path"])
    if scenario.vision_required and not scenario.external_assets:
        raise RuntimeError("Matching real visual golden assets are unavailable; synthetic pixels are not ground truth")
    return {"source": str(source)}


def embedding_files(model):
    """Look up existing cache entries only; never download or instantiate a model."""
    from huggingface_hub import try_to_load_from_cache

    def existing(filename):
        path = Path(model) / filename if Path(model).is_dir() else try_to_load_from_cache(model, filename)
        if isinstance(path, (str, Path)) and Path(path).is_file() and Path(path).stat().st_size:
            return Path(path)
        return None

    for filename in ("modules.json", "config.json", "tokenizer_config.json"):
        if not existing(filename):
            raise RuntimeError(f"Embedding model {model}: local file missing: {filename}")
    for module in json.loads(existing("modules.json").read_text(encoding="utf-8")):
        directory = module.get("path")
        configured = module.get("type", "").rsplit(".", 1)[-1] in {"Transformer", "Pooling", "Dense"}
        if directory and configured and not existing(f"{directory}/config.json"):
            raise RuntimeError(f"Embedding model {model}: local module configuration missing: {directory}")
    if not any(existing(name) for name in ("tokenizer.json", "vocab.txt", "spiece.model", "sentencepiece.bpe.model")):
        raise RuntimeError(f"Embedding model {model}: local tokenizer missing")
    if not any(existing(name) for name in ("model.safetensors", "pytorch_model.bin")):
        index = existing("model.safetensors.index.json") or existing("pytorch_model.bin.index.json")
        shards = set(json.loads(index.read_text(encoding="utf-8")).get("weight_map", {}).values()) if index else set()
        if not shards or not all(existing(name) for name in shards):
            raise RuntimeError(f"Embedding model {model}: local weights missing or incomplete")
    return {"version": model, "meaning": "Local files present; model not loaded"}


def trained_model_files(kind, path):
    """Allow the existing registry alternative without deserializing its model."""
    if Path(path).is_file() and Path(path).stat().st_size:
        return {"checkpoint": str(path), "meaning": "File present; model not loaded"}
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        require_file(path)
    from mlflow import MlflowClient

    from ml.anomaly.registry import REGISTERED_ANOMALY_MODEL_NAME
    from ml.registry import CANDIDATE_ALIAS, REGISTERED_MODEL_NAME

    name = REGISTERED_MODEL_NAME if kind == "triage" else REGISTERED_ANOMALY_MODEL_NAME
    version = MlflowClient(tracking_uri=uri).get_model_version_by_alias(name, CANDIDATE_ALIAS)
    return {"version": version.version, "source": "mlflow_candidate_metadata"}
