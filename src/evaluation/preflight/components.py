"""PostgreSQL, Qdrant and LLM endpoint probes."""

import os


class ComponentProbes:
    """Probe external services without starting or writing to them."""

    def probe_postgres(self):
        def postgres():
            from sqlalchemy import create_engine, text

            from persistence.database import DatabaseConfig, create_session_factory

            value = self.config.database_url or os.environ.get("DATABASE_URL")
            if not value:
                raise RuntimeError("DATABASE_URL is not configured; no test-database fallback is attempted")
            database = DatabaseConfig.from_url(value)
            engine = create_engine(
                database.url,
                pool_pre_ping=True,
                hide_parameters=True,
                connect_args={"connect_timeout": max(1, int(self.config.timeout_seconds))},
            )
            try:
                with engine.connect() as connection:
                    version = connection.execute(text("SHOW server_version")).scalar_one()
                    tables = connection.execute(text("SELECT to_regclass('claims')")).scalar_one()
                self.report.resources.update(database_engine=engine, session_factory=create_session_factory(engine))
            except Exception:
                engine.dispose()
                raise
            return {
                "version": version,
                "schema_available": tables is not None,
                "endpoint": database.url.render_as_string(hide_password=True),
            }

        self.probe("postgresql", postgres)

    def probe_qdrant(self):
        def qdrant():
            from qdrant_client import QdrantClient

            from retrieval.config import RetrievalConfig

            url = self.config.qdrant_url or RetrievalConfig.from_env().qdrant_url
            client = QdrantClient(
                url=url, api_key=os.environ.get("QDRANT_API_KEY"), timeout=self.config.timeout_seconds
            )
            try:
                available = [item.name for item in client.get_collections().collections]
                info = client.info()
            except Exception:
                client.close()
                raise
            self.report.resources.update(qdrant_client=client, collections=available)
            return {"version": info.version, "endpoint": url, "collections": available}

        self.probe("qdrant", qdrant, enabled=self.config.include_rag or self.config.include_similar)

    def probe_ollama(self):
        def ollama():
            import httpx

            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
            response = httpx.get(host + "/api/tags", timeout=self.config.timeout_seconds)
            response.raise_for_status()
            names = [item["name"] for item in response.json().get("models", [])]
            model = self.config.llm_model or os.environ.get("CLAIMS_LLM_MODEL")
            self.report.resources["ollama_models"] = names
            if not model:
                raise RuntimeError(
                    "Ollama reachable but CLAIMS_LLM_MODEL is not configured; installed models: " + ", ".join(names)
                )
            if model not in names and model + ":latest" not in names:
                raise RuntimeError(f"Requested Ollama model {model!r} is not installed; no pull attempted")
            self.report.resources["llm_model"] = model
            return {"version": model, "endpoint": host, "installed_models": names, "temperature": 0, "seed": 42}

        self.probe("ollama", ollama, enabled=self.config.include_agent or self.config.include_rag)

    def probe_mlflow(self):
        def mlflow():
            uri = os.environ.get("MLFLOW_TRACKING_URI")
            if not uri:
                raise RuntimeError(
                    "MLFLOW_TRACKING_URI is not configured; existing local trained artifacts are allowed"
                )
            if uri.startswith(("http://", "https://")):
                import httpx

                response = httpx.get(uri.rstrip("/") + "/health", timeout=self.config.timeout_seconds)
                response.raise_for_status()
            import mlflow as library

            library.set_tracking_uri(uri)
            return {"version": library.__version__, "tracking_uri": uri}

        self.probe("mlflow", mlflow)
