"""Runner."""

from __future__ import annotations

from typing import Any

from vision.damage.evaluation._params import EvaluationParams
from vision.damage.evaluation._runtime._inference import _evaluate_model
from vision.damage.evaluation._runtime._reports import _evaluation_payload, _save_evaluation
from vision.damage.evaluation._runtime._request import _prepare_request
from vision.damage.training import (
    DamageTrainingConfig,
    verify_class_names,
)


def run_evaluation(
    config: DamageTrainingConfig,
    *,
    params: EvaluationParams,
) -> dict[str, Any]:
    """
    Validator save_dir is observed rather than assumed, to catch auto-increment.
    """
    model_factory, environment_provider = params.model_factory, params.environment_provider
    request = _prepare_request(config, params)
    manifest, checkpoint = request.manifest, request.checkpoint
    evaluation_dir = config.output_dir / "evaluation"
    environment = environment_provider()
    model = model_factory(str(checkpoint))
    if getattr(model, "task", None) != "segment":
        raise ValueError("Selected checkpoint must have task=segment")
    verify_class_names(getattr(model, "names", None), manifest["category_mapping"])
    result = _evaluate_model(model, config, request.ap_confidence, evaluation_dir, environment)
    payload = _evaluation_payload(request, environment, result)
    _save_evaluation(request, model, payload)
    return payload
