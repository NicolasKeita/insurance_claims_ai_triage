"""Framework."""

from pathlib import Path
from types import SimpleNamespace

from vision.damage.training import (
    run_training,
)


def example_metric():
    group = SimpleNamespace(
        mp=0.7, mr=0.5, map50=0.6, map=0.4, ap_class_index=[1], class_result=lambda index: [0.7, 0.5, 0.6, 0.4]
    )
    return SimpleNamespace(box=group, seg=group)


class OrchestrationStub:
    """Simulates framework responses with bytes in tmp_path, never CV weights."""

    task = "segment"
    names = {0: "dent", 1: "scratch"}
    device = "cpu"

    def __init__(self):
        self.callbacks = {}
        self.training_kwargs = None
        self.validation_kwargs = None

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs):
        self.training_kwargs = kwargs
        best = Path(kwargs["project"]) / kwargs["name"] / "weights/best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"stub checkpoint for orchestration tests only")
        last = best.with_name("last.pt")
        last.write_bytes(b"different final stub checkpoint")
        self.trainer = SimpleNamespace(
            best=best,
            last=last,
            device="cpu",
            epoch=42,
            fitness=1.0,
            best_fitness=1.0,
            stopper=SimpleNamespace(best_epoch=43),
        )
        for callback in self.callbacks.get("on_model_save", []):
            callback(self.trainer)
        self.trainer.epoch = 99
        self.trainer.fitness = 0.5
        for callback in self.callbacks.get("on_model_save", []):
            callback(self.trainer)
        return example_metric()

    def val(self, **kwargs):
        self.validation_kwargs = kwargs
        result = example_metric()
        result.save_dir = Path(kwargs["project"]) / kwargs["name"]
        result.save_dir.mkdir(parents=True)
        return result


def completed_fixture(config, manifest):
    stub = OrchestrationStub()
    summary = run_training(
        config,
        model_factory=lambda _: stub,
        environment_provider=lambda: {"torch_version": "stub", "cuda_available": False},
        dataset_loader=lambda _: manifest,
    )
    return stub, summary
