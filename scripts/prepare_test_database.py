"""Preflight a dedicated test database using Alembic in a disposable schema."""

from pathlib import Path
import sys

# Test infrastructure is deliberately not part of the application package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from integration_support import (  # noqa: E402
    get_test_database_config, isolated_postgres_schema, migrate_test_schema,
)


def main() -> int:
    try:
        config = get_test_database_config()
        with isolated_postgres_schema(config) as engine:
            migrate_test_schema(engine)
    except Exception as error:
        print(f"PostgreSQL test preflight FAILED: {error}", file=sys.stderr)
        return 1
    print(
        "PostgreSQL test preflight PASSED: Alembic head and ORM metadata verified at "
        f"{config.url.render_as_string(hide_password=True)}; disposable schema cleaned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
