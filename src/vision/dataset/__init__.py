"""Legacy local COCO inspection/preparation and shared YOLO validation.

No dataset, framework, model or weight is loaded at module import. Inspection is
read-only. The COCO adapter preserves supplied splits and never changes sources.
The default HF mirror workflow uses its own frozen split in cardd_hf modules."""

from vision.dataset._constants import DEFAULT_MANIFEST_PATH as DEFAULT_MANIFEST_PATH
from vision.dataset._constants import DEFAULT_PREPARED_DIR as DEFAULT_PREPARED_DIR
from vision.dataset._constants import PREPARATION_VERSION as PREPARATION_VERSION
from vision.dataset._constants import PREPARED_MANIFEST_NAME as PREPARED_MANIFEST_NAME
from vision.dataset._constants import ROOT as ROOT
from vision.dataset._constants import SPLITS as SPLITS
from vision.dataset._files import _check_image as _check_image
from vision.dataset._files import _contained_file as _contained_file
from vision.dataset._files import _integer as _integer
from vision.dataset._files import _read_json as _read_json
from vision.dataset._files import _relative_filename as _relative_filename
from vision.dataset._files import _write_json as _write_json
from vision.dataset._files import annotation_hash as annotation_hash
from vision.dataset._models import CarddDataset as CarddDataset
from vision.dataset._models import CocoSplit as CocoSplit
from vision.dataset._models import DatasetValidationError as DatasetValidationError
from vision.dataset._models import _CocoConversion as _CocoConversion
from vision.dataset._models import _ConversionCounts as _ConversionCounts
from vision.dataset._models import _PreparedCounts as _PreparedCounts
from vision.dataset._polygons import _polygon as _polygon
from vision.dataset._polygons import _polygon_component as _polygon_component
from vision.dataset._polygons import (
    validate_segmentation_label as validate_segmentation_label,
)
from vision.dataset.coco._discovery import (
    discover_cardd_dataset as discover_cardd_dataset,
)
from vision.dataset.coco._discovery import (
    read_category_mapping as read_category_mapping,
)
from vision.dataset.coco._inspection import (
    build_dataset_manifest as build_dataset_manifest,
)
from vision.dataset.coco._inspection import (
    inspect_cardd_dataset as inspect_cardd_dataset,
)
from vision.dataset.coco._locations import _discover_path as _discover_path
from vision.dataset.coco._locations import _split_paths as _split_paths
from vision.dataset.coco._references import (
    _annotation_references as _annotation_references,
)
from vision.dataset.coco._references import _image_identities as _image_identities
from vision.dataset.preparation._annotations import (
    _convert_coco_annotations as _convert_coco_annotations,
)
from vision.dataset.preparation._conversion import (
    _convert_coco_split as _convert_coco_split,
)
from vision.dataset.preparation._conversion import (
    _convert_coco_splits as _convert_coco_splits,
)
from vision.dataset.preparation._conversion import (
    _update_coco_counts as _update_coco_counts,
)
from vision.dataset.preparation._converter import _coco_backend as _coco_backend
from vision.dataset.preparation._converter import (
    _try_ultralytics_conversion as _try_ultralytics_conversion,
)
from vision.dataset.preparation._images import _copy_coco_images as _copy_coco_images
from vision.dataset.preparation._preparation import (
    _preparation_manifest as _preparation_manifest,
)
from vision.dataset.preparation._preparation import (
    _preparation_output as _preparation_output,
)
from vision.dataset.preparation._preparation import (
    _validate_coco_stage as _validate_coco_stage,
)
from vision.dataset.preparation._preparation import (
    prepare_cardd_yolo as prepare_cardd_yolo,
)
from vision.dataset.prepared._entries import _prepared_entry as _prepared_entry
from vision.dataset.prepared._inventory import (
    _prepared_split_counts as _prepared_split_counts,
)
from vision.dataset.prepared._inventory import (
    _validate_prepared_splits as _validate_prepared_splits,
)
from vision.dataset.prepared._prepared import _prepared_header as _prepared_header
from vision.dataset.prepared._prepared import (
    _prepared_provenance as _prepared_provenance,
)
from vision.dataset.prepared._prepared import (
    load_prepared_dataset as load_prepared_dataset,
)
