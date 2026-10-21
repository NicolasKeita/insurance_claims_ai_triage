"""Support."""

from types import SimpleNamespace


class NativeMasks:
    def __init__(self, data):
        self.data = data

    @property
    def xyn(self):
        raise AssertionError("Native binary masks must not read lossy xyn contours")


def result(mask, *, names):
    return SimpleNamespace(
        names=names,
        orig_shape=(12, 20),
        masks=NativeMasks(mask[None]),
        boxes=SimpleNamespace(xyxyn=[[0.0, 0.0, 1.0, 1.0]], cls=[0], conf=[0.9]),
    )
