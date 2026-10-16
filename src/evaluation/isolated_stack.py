"""Explicit opt-in real PostgreSQL demo isolation; never targets application data.

The caller supplies the dedicated test URL. Each context owns one random schema,
applies the application's migrations, and drops only that schema at completion.
No fake model or test module participates in this real-stack setup.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from claims.case import ClaimCase
from claims.models import Claim
from persistence.database import create_session_factory, require_test_database
from persistence.models import Base
from persistence.repositories import ClaimRepository
from persistence.service import PersistenceService
from persistence.vision_assessments import VisionAssessmentRepository

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IsolatedPostgresStack:
    engine: Engine
    factory: sessionmaker
    schema: str


@contextmanager
def isolated_postgres(database_url) -> Iterator[IsolatedPostgresStack]:
    """Migrate a fresh schema in an explicitly validated ``*_test`` database."""
    config = require_test_database(database_url)
    schema = f"evaluation_e2e_{uuid4().hex}"
    admin = create_engine(config.url, hide_parameters=True,
                          connect_args={"connect_timeout": 5, "options": "-csearch_path=pg_catalog"})
    engine, created = None, False
    try:
        with admin.begin() as connection:
            actual = connection.scalar(text("SELECT current_database()"))
            if actual != config.url.database:
                raise RuntimeError("Evaluation setup refused: PostgreSQL database differs from the validated test URL")
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        created = True
        engine = create_engine(config.url, hide_parameters=True,
                               connect_args={"connect_timeout": 5, "options": f"-csearch_path={schema}"})
        migration = Config(str(ROOT / "alembic.ini"))
        migration.set_main_option("script_location", str(ROOT / "migrations"))
        with engine.begin() as connection:
            if inspect(connection).get_table_names():
                raise RuntimeError("Evaluation migrations require an empty isolated schema")
            migration.attributes["connection"] = connection
            command.upgrade(migration, "head")
            if not set(Base.metadata.tables).issubset(inspect(connection).get_table_names()):
                raise RuntimeError("Evaluation migrations did not create all application tables")
        yield IsolatedPostgresStack(engine, create_session_factory(engine), schema)
    finally:
        try:
            if engine is not None:
                engine.dispose()
        finally:
            try:
                if created:
                    with admin.begin() as connection:
                        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            finally:
                admin.dispose()


def seed_case(factory, case: ClaimCase | Claim, *, triage=None, anomaly=None, investigation=None, vision=None):
    """Import a controlled demo claim and append only actually computed outputs."""
    claim = case.claim if isinstance(case, ClaimCase) else case
    with factory() as session:
        existing = ClaimRepository(session).get_by_claim_id(claim.claim_id)
    service = PersistenceService(factory)
    if existing is None:
        service.import_claim(claim)
    elif existing != claim:
        raise ValueError(f"Controlled demo claim ID collision: {claim.claim_id}")
    saved = {}
    if any(value is not None for value in (triage, anomaly, investigation)):
        saved = service.save_assessments(claim.claim_id, triage=triage, anomaly=anomaly,
                                         investigation=investigation)
    if vision is not None:
        if vision.claim_id != claim.claim_id:
            raise ValueError("Visual assessment belongs to a different controlled claim")
        with factory.begin() as session:
            saved["vision"] = VisionAssessmentRepository(session).add(vision)
    return saved


def seed_history(factory, current: ClaimCase | Claim, fixtures):
    """Seed actual demo claim rows, including same-day/future/customer controls.

    Historical statistical labels are not fabricated. The current benchmark's
    fixtures request no prior anomalies, so missing assessments correctly yield
    zero previous anomaly flags in the real repository's aggregate.
    """
    current = current.claim if isinstance(current, ClaimCase) else current
    if any(fixture.get("is_anomalous", False) for fixture in fixtures):
        raise ValueError("Real evaluation cannot seed fake historical anomaly outputs; supply real assessments separately")
    claims = []
    for index, fixture in enumerate(fixtures):
        payload = current.model_dump(mode="json")
        payload["claim_id"] = fixture.get("claim_id", f"{current.claim_id}-HISTORY-{index}")
        payload["incident"]["date"] = (current.incident.date - timedelta(days=fixture.get("days_before", 10))).isoformat()
        payload["customer"]["customer_id"] = fixture.get("customer_id", current.customer.customer_id)
        payload["repair_estimate"] = {"amount": str(fixture.get("amount", 1000)),
                                      "currency": fixture.get("currency", current.repair_estimate.currency)}
        claim = Claim.model_validate(payload)
        seed_case(factory, claim)
        claims.append(claim)
    return claims
