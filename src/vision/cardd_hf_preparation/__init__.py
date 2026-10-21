"""Convert inspected FiftyOne mirror instances into validated YOLO segmentation.

This module does not import FiftyOne, contact the Hub, or load a model. Source
polygons remain instance polygons; multipart instances use the existing explicit
YOLO bridge policy. Images are copied once into an atomic prepared directory."""

from vision.cardd_hf_preparation._constants import (
    DEFAULT_HF_MANIFEST_PATH as DEFAULT_HF_MANIFEST_PATH,
)
from vision.cardd_hf_preparation._constants import (
    DEFAULT_HF_PREPARED_DIR as DEFAULT_HF_PREPARED_DIR,
)
from vision.cardd_hf_preparation._constants import (
    HF_PREPARATION_VERSION as HF_PREPARATION_VERSION,
)
from vision.cardd_hf_preparation._contracts import (
    _hf_source_contract as _hf_source_contract,
)
from vision.cardd_hf_preparation._contracts import (
    _hf_source_lookup as _hf_source_lookup,
)
from vision.cardd_hf_preparation._contracts import (
    _validate_hf_distribution as _validate_hf_distribution,
)
from vision.cardd_hf_preparation._fingerprint import _canonical_hash as _canonical_hash
from vision.cardd_hf_preparation._fingerprint import (
    hf_prepared_fingerprint as hf_prepared_fingerprint,
)
from vision.cardd_hf_preparation._manifest import _hf_manifest as _hf_manifest
from vision.cardd_hf_preparation._preparation import _hf_backend as _hf_backend
from vision.cardd_hf_preparation._preparation import (
    _validate_hf_stage as _validate_hf_stage,
)
from vision.cardd_hf_preparation._preparation import (
    prepare_cardd_hf_yolo as prepare_cardd_hf_yolo,
)
from vision.cardd_hf_preparation._validation import (
    _validate_hf_entry as _validate_hf_entry,
)
from vision.cardd_hf_preparation._validation import (
    validate_hf_prepared_manifest as validate_hf_prepared_manifest,
)
from vision.cardd_hf_preparation.conversion._context import (
    _HfConversion as _HfConversion,
)
from vision.cardd_hf_preparation.conversion._context import (
    _HfSplitInventory as _HfSplitInventory,
)
from vision.cardd_hf_preparation.conversion._conversion import (
    _convert_hf_sample as _convert_hf_sample,
)
from vision.cardd_hf_preparation.conversion._conversion import (
    _convert_hf_split as _convert_hf_split,
)
from vision.cardd_hf_preparation.conversion._conversion import (
    _convert_hf_splits as _convert_hf_splits,
)
from vision.cardd_hf_preparation.conversion._conversion import _hf_rows as _hf_rows
from vision.cardd_hf_preparation.conversion._files import _hf_output as _hf_output
from vision.cardd_hf_preparation.conversion._files import (
    _write_hf_sample as _write_hf_sample,
)
from vision.cardd_hf_preparation.conversion._geometry import (
    _hf_annotation as _hf_annotation,
)
from vision.cardd_hf_preparation.conversion._geometry import _hf_polygon as _hf_polygon
