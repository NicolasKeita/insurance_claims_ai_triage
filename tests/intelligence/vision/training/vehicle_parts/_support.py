"""Support."""

from pathlib import Path
from types import SimpleNamespace


class PartStub:
    task = "segment"
    names = {0: "hood"}

    def __init__(self):
        self.callbacks = {}

    def add_callback(self, event, callback):
        self.callbacks.setdefault(event, []).append(callback)

    def train(self, **kwargs):
        self.train_arguments = kwargs
        best = Path(kwargs["project"]) / "train/weights/best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"parts-selected-on-VAL-fixture")
        last = best.with_name("last.pt")
        last.write_bytes(b"parts-last-fixture")
        self.trainer = SimpleNamespace(
            best=best,
            last=last,
            device="cpu",
            epoch=99,
            fitness=0.4,
            best_fitness=0.4,
            stopper=SimpleNamespace(best_epoch=100),
        )
        for callback in self.callbacks["on_model_save"]:
            callback(self.trainer)
        return SimpleNamespace()

    def val(self, **kwargs):
        self.val_arguments = kwargs
        for callback in self.callbacks.get("on_val_start", []):
            callback(SimpleNamespace(device="cpu"))
        directory = Path(kwargs["project"]) / "evaluation"
        directory.mkdir()
        return SimpleNamespace(save_dir=directory)
