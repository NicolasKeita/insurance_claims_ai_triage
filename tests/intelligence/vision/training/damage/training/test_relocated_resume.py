"""Resume a relocated run without rewriting its checkpoint or frozen provenance."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

from persistence.artifact_storage import relocate_path
from tests.intelligence.vision.training.damage._dataset import offline_environment as offline_environment
from tests.intelligence.vision.training.damage._dataset import prepared as prepared
from tests.intelligence.vision.training.damage._resume import ResumeStub, interrupted_fixture
from vision.damage.training import run_training


def test_relocated_run_resumes_without_rewriting_its_saved_identity(prepared, tmp_path, monkeypatch):
    from vision.damage.training import _records, _resume

    config, manifest = prepared
    original_vision = tmp_path / "artifacts/vision"
    original = replace(config, output_dir=original_vision / "interrupted_run")
    metadata = interrupted_fixture(original, manifest)
    stub = ResumeStub(original, metadata)
    frozen_checkpoint = deepcopy(stub.ckpt)
    recovery_name = "recovery_training_metadata.json"
    frozen_metadata = (original.output_dir / recovery_name).read_bytes()
    destination = tmp_path / "artifacts/Modeles/vision"
    destination.parent.mkdir()
    original_vision.rename(destination)
    current = replace(config, output_dir=destination / "interrupted_run")
    stub.config = current
    resolve = partial(relocate_path, root=tmp_path)
    monkeypatch.setattr(_resume, "relocate_path", resolve)
    monkeypatch.setattr(_records, "relocate_path", resolve)
    loaded_metadata = _resume.load_resume_metadata(current, manifest, 0.25)
    _resume.validate_resume_checkpoint(stub, current, loaded_metadata, manifest)
    assert (current.output_dir / recovery_name).read_bytes() == frozen_metadata
    assert stub.ckpt == frozen_checkpoint

    summary = run_training(
        current,
        model_factory=lambda _: stub,
        environment_provider=lambda: {},
        dataset_loader=lambda _: manifest,
        resume=True,
    )

    assert summary["resumed"] is True
    assert summary["base_checkpoint"] == str(destination / "original_pretrained.pt")
    assert summary["base_checkpoint_sha256"] == metadata["base_checkpoint_sha256"]
    assert stub.ckpt == frozen_checkpoint
