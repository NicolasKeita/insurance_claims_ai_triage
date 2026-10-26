"""Read-only sessions and optional stored perception for repository fixtures."""

import agent.tools._visual as visual_adapters


class ReadSessionFactory:
    def __init__(self):
        self.open_count = 0

    def __call__(self):
        return self

    def __enter__(self):
        self.open_count += 1
        return self

    def __exit__(self, *_):
        self.open_count -= 1

    def add(self, *_):
        raise AssertionError("Business writes are forbidden")

    flush = commit = add


def install_visual_repository(monkeypatch, calls, records):
    class VisualAssessments:
        def __init__(self, session):
            assert session.open_count == 1

        def latest_for_claim(self, claim_id):
            calls.append(("vision", claim_id))
            return records.get("vision")

    monkeypatch.setattr(visual_adapters, "VisionAssessmentRepository", VisualAssessments)
