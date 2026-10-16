"""Read-only dependency probes for the explicit, opt-in real evaluation.

Heavy adapters are loaded once and retained for the run. No downloaded models,
service startup, writes to the database/vector stores, or fake replacements.
"""

from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
from time import perf_counter
import tomllib
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/evaluation/e2e_v1.toml"


@dataclass(frozen=True)
class RealEvaluationConfig:
    include_vision: bool = True
    include_agent: bool = True
    include_rag: bool = True
    include_similar: bool = True
    fail_fast: bool = False
    dependency_unavailable: str = "SKIP"
    timeout_seconds: float = 2.0
    database_url: str | None = None
    qdrant_url: str | None = None
    llm_model: str | None = None
    triage_checkpoint: Path = ROOT / "artifacts/triage_v1/logistic_regression.joblib"
    anomaly_checkpoint: Path = ROOT / "artifacts/anomaly_v1/anomaly_detector.joblib"
    damage_checkpoint: Path = ROOT / "artifacts/vision/damage_segmentation_v1/train/weights/best.pt"
    part_checkpoint: Path = ROOT / "artifacts/vision/vehicle_parts_v1/train/weights/best.pt"
    vision_device: str | None = None
    source_summary: Path = ROOT / "artifacts/vision/damage_segmentation_v1/test_smoke/smoke_summary.json"
    training_summary: Path = ROOT / "artifacts/vision/damage_segmentation_v1/training_summary.json"
    diagnostic_max_images: int = 2
    synthetic_declared_damage: tuple[str, ...] = ("FRONT_BUMPER",)
    synthetic_quoted_damage: tuple[str, ...] = ()
    fixture_dir: Path = ROOT / "artifacts/evaluation/e2e_v1/fixtures"
    isolated_demo_stack: bool = False

    def __post_init__(self):
        if self.dependency_unavailable not in {"SKIP", "FAIL"}:
            raise ValueError("dependency_unavailable must be SKIP or FAIL")
        if not 0 < self.timeout_seconds <= 30:
            raise ValueError("timeout_seconds must be positive and at most 30")
        if self.diagnostic_max_images < 1:
            raise ValueError("diagnostic_max_images must be positive")

    @classmethod
    def from_toml(cls, path: Path = DEFAULT_CONFIG, **overrides):
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("benchmark_version") != "e2e_claims_v1":
            raise ValueError("Unsupported real evaluation benchmark_version")
        models, visual = raw.get("models", {}), raw.get("vision_diagnostic", {})
        values = {key: raw[key] for key in (
            "include_vision", "include_agent", "include_rag", "include_similar",
            "fail_fast", "dependency_unavailable", "timeout_seconds",
        ) if key in raw}
        for key in ("triage_checkpoint", "anomaly_checkpoint", "damage_checkpoint", "part_checkpoint"):
            if key in models:
                model_path = Path(models[key]).expanduser()
                values[key] = model_path if model_path.is_absolute() else ROOT / model_path
        # Current project environment settings take precedence over versioned defaults.
        for key, variable in (("damage_checkpoint", "DAMAGE_MODEL_PATH"), ("part_checkpoint", "PART_MODEL_PATH")):
            if os.environ.get(variable):
                values[key] = Path(os.environ[variable]).expanduser()
        values["vision_device"] = os.environ.get("DAMAGE_DEVICE") or models.get("vision_device")
        for key in ("source_summary", "training_summary"):
            if key in visual:
                source = Path(visual[key]).expanduser()
                values[key] = source if source.is_absolute() else ROOT / source
        values["diagnostic_max_images"] = int(visual.get("max_images", 2))
        values["synthetic_declared_damage"] = tuple(visual.get("synthetic_declared_damage", ["FRONT_BUMPER"]))
        values["synthetic_quoted_damage"] = tuple(visual.get("synthetic_quoted_damage", []))
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


@dataclass
class PreflightReport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict, repr=False)

    def available(self, name: str) -> bool:
        return any(row["component"] == name and row["status"] == "PASS" for row in self.rows)

    def reason(self, name: str) -> str:
        return next((row.get("reason", "") for row in self.rows if row["component"] == name), "Not preflighted")

    def close(self):
        engine = self.resources.get("database_engine")
        if engine is not None:
            engine.dispose()
        client = self.resources.get("qdrant_client")
        if client is not None:
            client.close()


def checkpoint_version(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def run_preflight(config: RealEvaluationConfig, *, progress: Callable[[dict], None] | None = None) -> PreflightReport:
    report = PreflightReport()

    def probe(name, operation, *, enabled=True):
        start = perf_counter()
        row = {"component": name, "status": "SKIP", "version": None, "reason": "Disabled by configuration"}
        if enabled:
            try:
                metadata = operation() or {}
                row.update(status="PASS", reason=None, **metadata)
            except Exception as error:
                # Missing optional dependencies are explicit SKIP/FAIL according to config.
                row.update(status="SKIP" if name == "mlflow" else config.dependency_unavailable,
                           reason=f"{type(error).__name__}: {error}")
        row["duration_seconds"] = round(perf_counter() - start, 6)
        report.rows.append(row)
        if progress is not None:
            progress(row)
        return row

    def postgres():
        from sqlalchemy import create_engine, text
        from persistence.database import DatabaseConfig, create_session_factory
        value = config.database_url or os.environ.get("DATABASE_URL")
        if not value:
            raise RuntimeError("DATABASE_URL is not configured; no test-database fallback is attempted")
        database = DatabaseConfig.from_url(value)
        engine = create_engine(database.url, pool_pre_ping=True, hide_parameters=True,
                               connect_args={"connect_timeout": max(1, int(config.timeout_seconds))})
        try:
            with engine.connect() as connection:
                version = connection.execute(text("SHOW server_version")).scalar_one()
                tables = connection.execute(text("SELECT to_regclass('claims')")).scalar_one()
            report.resources.update(database_engine=engine, session_factory=create_session_factory(engine))
        except Exception:
            engine.dispose()
            raise
        return {"version": version, "schema_available": tables is not None,
                "endpoint": database.url.render_as_string(hide_password=True)}

    probe("postgresql", postgres)

    def qdrant():
        from qdrant_client import QdrantClient
        from retrieval.config import RetrievalConfig
        url = config.qdrant_url or RetrievalConfig.from_env().qdrant_url
        client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=config.timeout_seconds)
        try:
            available = [item.name for item in client.get_collections().collections]
            info = client.info()
        except Exception:
            client.close()
            raise
        report.resources.update(qdrant_client=client, collections=available)
        return {"version": info.version, "endpoint": url, "collections": available}

    probe("qdrant", qdrant, enabled=config.include_rag or config.include_similar)
    from knowledge.config import KnowledgeConfig
    from retrieval.config import RetrievalConfig
    knowledge_config = KnowledgeConfig.from_env()
    retrieval_config = RetrievalConfig.from_env()
    if config.qdrant_url:
        knowledge_config = replace(knowledge_config, qdrant_url=config.qdrant_url)
        retrieval_config = replace(retrieval_config, qdrant_url=config.qdrant_url)
    report.resources.update(knowledge_config=knowledge_config, retrieval_config=retrieval_config)

    def collections(names):
        if not report.available("qdrant"):
            raise RuntimeError("Qdrant is unavailable")
        missing = set(names) - set(report.resources["collections"])
        if missing:
            raise RuntimeError("Missing collections: " + ", ".join(sorted(missing)))
        return {"collections": list(names), "point_counts": {
            name: report.resources["qdrant_client"].get_collection(name).points_count for name in names}}

    probe("knowledge_collections", lambda: collections([knowledge_config.policy_collection, knowledge_config.procedure_collection]),
          enabled=config.include_rag)
    probe("similar_claims_collection", lambda: collections([retrieval_config.collection]), enabled=config.include_similar)

    def embeddings():
        from retrieval.embeddings import SentenceTransformerEmbeddingProvider
        # local_files_only is enforced even when another import initialized HF settings.
        provider = SentenceTransformerEmbeddingProvider(knowledge_config.embedding_model, asymmetric=True, local_files_only=True)
        dimension = provider.dimension
        report.resources["embeddings"] = provider
        return {"version": provider.model_name, "dimension": dimension, "local_files_only": True}

    probe("embeddings", embeddings, enabled=config.include_rag or config.include_similar)

    def ollama():
        import httpx
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        response = httpx.get(host + "/api/tags", timeout=config.timeout_seconds)
        response.raise_for_status()
        names = [item["name"] for item in response.json().get("models", [])]
        model = config.llm_model or os.environ.get("CLAIMS_LLM_MODEL")
        report.resources["ollama_models"] = names
        if not model:
            raise RuntimeError("Ollama reachable but CLAIMS_LLM_MODEL is not configured; installed models: " + ", ".join(names))
        if model not in names and model + ":latest" not in names:
            raise RuntimeError(f"Requested Ollama model {model!r} is not installed; no pull attempted")
        report.resources["llm_model"] = model
        return {"version": model, "endpoint": host, "installed_models": names, "temperature": 0, "seed": 42}

    probe("ollama", ollama, enabled=config.include_agent or config.include_rag)

    def mlflow():
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            raise RuntimeError("MLFLOW_TRACKING_URI is not configured; existing local trained artifacts are allowed")
        if uri.startswith(("http://", "https://")):
            import httpx
            response = httpx.get(uri.rstrip("/") + "/health", timeout=config.timeout_seconds)
            response.raise_for_status()
        import mlflow as library
        library.set_tracking_uri(uri)
        return {"version": library.__version__, "tracking_uri": uri}

    probe("mlflow", mlflow)

    def trained_model(kind, path):
        registry_error = None
        if report.available("mlflow"):
            try:
                if kind == "triage":
                    from ml.inference import load_candidate_predictor
                    predictor = load_candidate_predictor()
                else:
                    from ml.anomaly_inference import load_candidate_anomaly_predictor
                    predictor = load_candidate_anomaly_predictor()
                report.resources[kind] = predictor
                report.resources[kind + "_source"] = "mlflow_candidate"
                return {"version": predictor.version, "source": "mlflow_candidate", "alias": predictor.alias}
            except Exception as error:
                registry_error = f"{type(error).__name__}: {error}"
        import joblib
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Existing trained checkpoint unavailable: {path}; registry: {registry_error}")
        model = joblib.load(path)
        if kind == "triage":
            if not hasattr(model, "predict_proba") or "classifier" not in model.named_steps:
                raise ValueError("Triage checkpoint does not implement the existing trained Pipeline contract")
            from ml.inference import RegisteredTriagePredictor
            predictor = RegisteredTriagePredictor(model, "local:" + path.name, checkpoint_version(path), "local-artifact")
        else:
            from ml.anomaly_training import AnomalyDetector
            if not isinstance(model, AnomalyDetector):
                raise ValueError("Anomaly checkpoint is not the project AnomalyDetector")
            predictor = model
        report.resources[kind] = predictor
        report.resources[kind + "_source"] = "existing_local_trained_artifact"
        return {"version": checkpoint_version(path), "source": "existing_local_trained_artifact",
                "checkpoint": str(path.resolve()), "registry_unavailable_reason": registry_error,
                "fallback_reason": None if report.available("mlflow") else report.reason("mlflow")}

    probe("triage", lambda: trained_model("triage", config.triage_checkpoint))
    probe("anomaly", lambda: trained_model("anomaly", config.anomaly_checkpoint))

    def detector(kind):
        from vision.config import VisionSettings
        from vision.detector import UltralyticsDamageDetector
        from vision.parts_detector import UltralyticsVehiclePartDetector, VehiclePartSettings
        if kind == "damage":
            adapter = UltralyticsDamageDetector(VisionSettings(
                model_path=config.damage_checkpoint, device=config.vision_device))
        else:
            adapter = UltralyticsVehiclePartDetector(VehiclePartSettings(
                model_path=config.part_checkpoint, device=config.vision_device))
        # Validate loadability, segmentation task and complete domain taxonomy.
        model = adapter._load_model()
        report.resources[kind + "_detector"] = adapter
        return {"version": adapter.model_metadata.version,
                "checkpoint": adapter.model_metadata.identifier, "task": model.task, "class_count": len(model.names)}

    probe("damage_vision", lambda: detector("damage"), enabled=config.include_vision)
    probe("part_vision", lambda: detector("part"), enabled=config.include_vision)
    return report


def print_preflight_row(row: dict) -> None:
    print(f"{row['component']:<27} {row['status']:<5} {row.get('version') or '-'} "
          f"{row.get('reason') or ''}", flush=True)
