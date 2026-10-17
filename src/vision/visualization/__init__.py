"""Debug rendering from domain observations, never an application data source."""

from vision.visualization._damage import save_annotated_image
from vision.visualization._multimodal import save_multimodal_annotated_image
from vision.visualization._palette import COLORS, _paint_polygon

__all__ = ["COLORS", "_paint_polygon", "save_annotated_image", "save_multimodal_annotated_image"]
