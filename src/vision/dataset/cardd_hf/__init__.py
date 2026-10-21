"""Inspect the CarDD HF mirror through FiftyOne, without eager dependencies.

This source is a subset/mirror, never the official CarDD train/val/test splits.
The normalized contract contains real instance geometry only: no bbox fallback."""

from vision.dataset.cardd_hf._constants import _LOAD_LOCK as _LOAD_LOCK
from vision.dataset.cardd_hf._constants import DEFAULT_HF_REPOSITORY as DEFAULT_HF_REPOSITORY
from vision.dataset.cardd_hf._constants import Point as Point
from vision.dataset.cardd_hf._constants import Polygon as Polygon
from vision.dataset.cardd_hf._inspection import inspect_cardd_hf as inspect_cardd_hf
from vision.dataset.cardd_hf._loading import _official_load as _official_load
from vision.dataset.cardd_hf._loading import _PinnedHub as _PinnedHub
from vision.dataset.cardd_hf._loading import _resolve_revision as _resolve_revision
from vision.dataset.cardd_hf._loading import default_hf_cache_dir as default_hf_cache_dir
from vision.dataset.cardd_hf._loading import load_cardd_hf as load_cardd_hf
from vision.dataset.cardd_hf._models import CarddHfConfig as CarddHfConfig
from vision.dataset.cardd_hf._models import HfDamageDataset as HfDamageDataset
from vision.dataset.cardd_hf._models import HfDamageInstance as HfDamageInstance
from vision.dataset.cardd_hf._models import HfDamageSample as HfDamageSample
from vision.dataset.cardd_hf._normalization import _normalize_samples as _normalize_samples
from vision.dataset.cardd_hf._normalization import (
    normalize_fiftyone_dataset as normalize_fiftyone_dataset,
)
from vision.dataset.cardd_hf.annotations._fields import _get as _get
from vision.dataset.cardd_hf.annotations._fields import _kind as _kind
from vision.dataset.cardd_hf.annotations._fields import _label as _label
from vision.dataset.cardd_hf.annotations._fields import _schema_kind as _schema_kind
from vision.dataset.cardd_hf.annotations._instances import (
    _detection_geometry as _detection_geometry,
)
from vision.dataset.cardd_hf.annotations._instances import _instances as _instances
from vision.dataset.cardd_hf.annotations._labels import _collect_field as _collect_field
from vision.dataset.cardd_hf.annotations._schema import _annotation_schema as _annotation_schema
from vision.dataset.cardd_hf.annotations._schema import _inspect_fields as _inspect_fields
from vision.dataset.cardd_hf.annotations._schema import _select_field as _select_field
from vision.dataset.cardd_hf.masks._contours import _bridge_holes as _bridge_holes
from vision.dataset.cardd_hf.masks._contours import _trace_rings as _trace_rings
from vision.dataset.cardd_hf.masks._geometry import _area as _area
from vision.dataset.cardd_hf.masks._geometry import _bridge as _bridge
from vision.dataset.cardd_hf.masks._geometry import _contains as _contains
from vision.dataset.cardd_hf.masks._geometry import _polygon as _polygon
from vision.dataset.cardd_hf.masks._raster import _binary_mask as _binary_mask
from vision.dataset.cardd_hf.masks._raster import _boundary_edges as _boundary_edges
from vision.dataset.cardd_hf.masks._raster import _mask_polygons as _mask_polygons
