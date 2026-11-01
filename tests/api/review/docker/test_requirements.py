"""The Docker archive image uses existing constraints without model packages."""

from pathlib import Path

import pytest
import tomllib

from deploy.docker.review_requirements import review_requirements


def test_runtime_constraints_come_from_application_source_without_model_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    selected = review_requirements(project)
    assert len(selected) == 3
    assert set(selected) <= set(project["project"]["dependencies"])
    assert any(item.startswith("fastapi[standard]") for item in selected)
    assert any(item.startswith("pydantic") for item in selected)
    assert any(item.startswith("SQLAlchemy") for item in selected)
    assert not any("torch" in item or "mlflow" in item or "sentence-transformers" in item for item in selected)


@pytest.mark.parametrize(
    "requirements",
    [
        ["fastapi[standard]>=1", "pydantic>=2"],
        ["fastapi[standard]>=1", "pydantic>=2", "SQLAlchemy>=2", "sqlalchemy<3"],
        ["fastapi>=1", "pydantic>=2", "SQLAlchemy>=2"],
    ],
)
def test_missing_ambiguous_or_serverless_dependency_configuration_fails(requirements):
    with pytest.raises(ValueError):
        review_requirements({"project": {"dependencies": requirements}})
