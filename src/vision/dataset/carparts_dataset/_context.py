"""State of a read-only Carparts inventory."""


class _Inspection:
    """Read-only dataset inventory and diagnostics shared across supplied splits."""

    def __init__(self, root, mapping, report):
        self.root, self.mapping, self.report = root, mapping, report
        self.seen_images = set()
