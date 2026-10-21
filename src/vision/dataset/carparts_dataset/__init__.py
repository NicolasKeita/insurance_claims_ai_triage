"Official Carparts-Seg preparation and read-only inspection; no import downloads."

from vision.dataset.carparts_dataset._constants import (
    DEFAULT_DATASET_ROOT as DEFAULT_DATASET_ROOT,
)
from vision.dataset.carparts_dataset._constants import (
    DEFAULT_DATASET_YAML as DEFAULT_DATASET_YAML,
)
from vision.dataset.carparts_dataset._constants import DEFAULT_MANIFEST as DEFAULT_MANIFEST
from vision.dataset.carparts_dataset._constants import IMAGE_SUFFIXES as IMAGE_SUFFIXES
from vision.dataset.carparts_dataset._constants import MANIFEST_NAME as MANIFEST_NAME
from vision.dataset.carparts_dataset._constants import OFFICIAL_ARCHIVE as OFFICIAL_ARCHIVE
from vision.dataset.carparts_dataset._constants import OFFICIAL_SOURCE as OFFICIAL_SOURCE
from vision.dataset.carparts_dataset._constants import (
    PREPARATION_VERSION as PREPARATION_VERSION,
)
from vision.dataset.carparts_dataset._context import _Inspection as _Inspection
from vision.dataset.carparts_dataset._inspection import (
    _inspection_metadata as _inspection_metadata,
)
from vision.dataset.carparts_dataset._inspection import (
    _source_overlap_audit as _source_overlap_audit,
)
from vision.dataset.carparts_dataset._inspection import (
    inspect_carparts_dataset as inspect_carparts_dataset,
)
from vision.dataset.carparts_dataset._inventory import (
    _inspect_carparts_image as _inspect_carparts_image,
)
from vision.dataset.carparts_dataset._inventory import (
    _inspect_carparts_split as _inspect_carparts_split,
)
from vision.dataset.carparts_dataset._inventory import (
    _inspect_carparts_splits as _inspect_carparts_splits,
)
from vision.dataset.carparts_dataset._preparation import (
    _carparts_manifest as _carparts_manifest,
)
from vision.dataset.carparts_dataset._preparation import (
    _normalize_carparts_yaml as _normalize_carparts_yaml,
)
from vision.dataset.carparts_dataset._preparation import (
    _official_carparts as _official_carparts,
)
from vision.dataset.carparts_dataset._preparation import (
    prepare_carparts_dataset as prepare_carparts_dataset,
)
from vision.dataset.carparts_dataset._provenance import (
    carparts_provenance as carparts_provenance,
)
from vision.dataset.carparts_dataset._provenance import (
    load_carparts_dataset as load_carparts_dataset,
)
from vision.dataset.carparts_dataset._schema import _class_mapping as _class_mapping
from vision.dataset.carparts_dataset._schema import _read_yaml as _read_yaml
from vision.dataset.carparts_dataset._schema import _require_valid as _require_valid
