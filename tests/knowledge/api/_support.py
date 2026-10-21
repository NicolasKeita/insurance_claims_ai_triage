"""Support."""

from knowledge.models import GroundedAnswer


class FakeSearch:
    def __init__(self):
        self.calls = []
        self.closed = False

    def search(self, *args, **kwargs):
        self.calls.append(kwargs)
        return []

    def close(self):
        self.closed = True


class FakeRag:
    def __init__(self):
        self.calls = []

    def answer(self, *args, **kwargs):
        self.calls.append(kwargs)
        return GroundedAnswer(answer="Demo excerpts are insufficient.", citations=[], insufficient_evidence=True)
