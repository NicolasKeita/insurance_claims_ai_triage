"""Trained ML checkpoint and vision detector probes."""

from evaluation.preflight.report import checkpoint_version


class ModelProbes:
    """Validate existing local artifacts without downloading or pulling anything."""

    def probe_models(self):
        self.probe("triage", lambda: self.trained_model("triage", self.config.triage_checkpoint))
        self.probe("anomaly", lambda: self.trained_model("anomaly", self.config.anomaly_checkpoint))

    def trained_model(self, kind, path):
        registry_error = None
        if self.report.available("mlflow"):
            try:
                return self.registry_model(kind)
            except Exception as error:
                registry_error = f"{type(error).__name__}: {error}"
        return self.checkpoint_model(kind, path, registry_error)

    def registry_model(self, kind):
        if kind == "triage":
            from ml.inference import load_candidate_predictor

            predictor = load_candidate_predictor()
        else:
            from ml.anomaly_inference import load_candidate_anomaly_predictor

            predictor = load_candidate_anomaly_predictor()
        self.report.resources[kind] = predictor
        self.report.resources[kind + "_source"] = "mlflow_candidate"
        return {"version": predictor.version, "source": "mlflow_candidate", "alias": predictor.alias}

    def checkpoint_model(self, kind, path, registry_error):
        import joblib

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Existing trained checkpoint unavailable: {path}; registry: {registry_error}")
        predictor = self._validated_checkpoint(kind, joblib.load(path), path)
        self.report.resources[kind] = predictor
        self.report.resources[kind + "_source"] = "existing_local_trained_artifact"
        return {
            "version": checkpoint_version(path),
            "source": "existing_local_trained_artifact",
            "checkpoint": str(path.resolve()),
            "registry_unavailable_reason": registry_error,
            "fallback_reason": None if self.report.available("mlflow") else self.report.reason("mlflow"),
        }

    @staticmethod
    def _validated_checkpoint(kind, model, path):
        if kind == "triage":
            if not hasattr(model, "predict_proba") or "classifier" not in model.named_steps:
                raise ValueError("Triage checkpoint does not implement the existing trained Pipeline contract")
            from ml.inference import RegisteredTriagePredictor

            return RegisteredTriagePredictor(
                model, "local:" + path.name, checkpoint_version(path), "local-artifact"
            )
        from ml.anomaly_training import AnomalyDetector

        if not isinstance(model, AnomalyDetector):
            raise ValueError("Anomaly checkpoint is not the project AnomalyDetector")
        return model

    def probe_detectors(self):
        self.probe("damage_vision", lambda: self.detector("damage"), enabled=self.config.include_vision)
        self.probe("part_vision", lambda: self.detector("part"), enabled=self.config.include_vision)

    def detector(self, kind):
        """Validate loadability, segmentation task and complete domain taxonomy."""
        from vision.config import VisionSettings
        from vision.detector import UltralyticsDamageDetector
        from vision.parts_detector import UltralyticsVehiclePartDetector, VehiclePartSettings

        if kind == "damage":
            adapter = UltralyticsDamageDetector(
                VisionSettings(model_path=self.config.damage_checkpoint, device=self.config.vision_device)
            )
        else:
            adapter = UltralyticsVehiclePartDetector(
                VehiclePartSettings(model_path=self.config.part_checkpoint, device=self.config.vision_device)
            )
        model = adapter._load_model()
        self.report.resources[kind + "_detector"] = adapter
        return {
            "version": adapter.model_metadata.version,
            "checkpoint": adapter.model_metadata.identifier,
            "task": model.task,
            "class_count": len(model.names),
        }