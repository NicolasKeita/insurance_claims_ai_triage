"""Shared configuration and fixed dataset contracts."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test"

DEFAULT_TEST_QDRANT_URL = "http://127.0.0.1:6335"
