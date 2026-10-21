"""Frozen, portable, leakage-checked splits for the CarDD Hugging Face mirror.

This is a project split, not an official CarDD benchmark split.  The small
stratifier uses image-level multilabels and keeps byte-identical images together.
No FiftyOne, Hugging Face or training framework is imported by this module."""

from vision.cardd_hf_split._constants import ALGORITHM as ALGORITHM
from vision.cardd_hf_split._constants import DEFAULT_RATIOS as DEFAULT_RATIOS
from vision.cardd_hf_split._constants import (
    DEFAULT_SPLIT_MANIFEST_PATH as DEFAULT_SPLIT_MANIFEST_PATH,
)
from vision.cardd_hf_split._constants import SPLIT_VERSION as SPLIT_VERSION
from vision.cardd_hf_split._constants import SPLITS as SPLITS
from vision.cardd_hf_split._constants import (
    SUPPORTED_SPLIT_ALGORITHMS as SUPPORTED_SPLIT_ALGORITHMS,
)
from vision.cardd_hf_split._identity import _counts as _counts
from vision.cardd_hf_split._identity import _digest as _digest
from vision.cardd_hf_split._identity import _Group as _Group
from vision.cardd_hf_split._identity import _parameters as _parameters
from vision.cardd_hf_split._manifest import _new_split_manifest as _new_split_manifest
from vision.cardd_hf_split._manifest import (
    create_or_load_cardd_hf_split as create_or_load_cardd_hf_split,
)
from vision.cardd_hf_split._metadata import (
    _distribution_warnings as _distribution_warnings,
)
from vision.cardd_hf_split._metadata import _split_metadata as _split_metadata
from vision.cardd_hf_split._validation import (
    _validate_assignments as _validate_assignments,
)
from vision.cardd_hf_split._validation import (
    _validate_split_fingerprint as _validate_split_fingerprint,
)
from vision.cardd_hf_split._validation import (
    validate_cardd_hf_split as validate_cardd_hf_split,
)
from vision.cardd_hf_split.allocation._balance import _balance_groups as _balance_groups
from vision.cardd_hf_split.allocation._balance import (
    _improve_balance as _improve_balance,
)
from vision.cardd_hf_split.allocation._balance import _pattern_pairs as _pattern_pairs
from vision.cardd_hf_split.allocation._greedy import _Allocation as _Allocation
from vision.cardd_hf_split.allocation._greedy import _assign as _assign
from vision.cardd_hf_split.allocation._packing import _packing_exists as _packing_exists
from vision.cardd_hf_split.allocation._packing import _packing_search as _packing_search
from vision.cardd_hf_split.grouping._samples import (
    _sample_annotations as _sample_annotations,
)
from vision.cardd_hf_split.grouping._samples import _validate_sample as _validate_sample
from vision.cardd_hf_split.grouping._source import _source_context as _source_context
from vision.cardd_hf_split.grouping._source import _source_groups as _source_groups
