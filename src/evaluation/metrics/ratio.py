"""Explicit ratio helper that never hides a missing denominator."""


def ratio(numerator, denominator):
    """Return None when nothing was executed instead of inventing a rate."""
    return numerator / denominator if denominator else None