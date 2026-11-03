"""Local artifact organization and compatibility with frozen historical paths."""

from persistence.artifact_storage.layout import LEGACY_LOCATIONS, ROOT, relocate_override, relocate_path

__all__ = ["LEGACY_LOCATIONS", "ROOT", "relocate_override", "relocate_path"]
