"""Official Carparts-Seg source locations and defaults."""

from vision.training import REPO_ROOT

PREPARATION_VERSION = "carparts_official_yolo_segmentation_v1"
OFFICIAL_SOURCE = "https://docs.ultralytics.com/datasets/segment/carparts-seg/"
OFFICIAL_ARCHIVE = "https://github.com/ultralytics/assets/releases/download/v0.0.0/carparts-seg.zip"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data/external/carparts-seg"
DEFAULT_DATASET_YAML = DEFAULT_DATASET_ROOT / "dataset.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "artifacts/vision/carparts_dataset_manifest.json"
MANIFEST_NAME = "carparts_dataset_manifest.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONFIG_PATH = "configs/vision/carparts_yolo26n_seg_v1.toml"
METRICS_SCOPE = "Official Carparts-Seg supplied TEST; TRAIN/VAL used for optimization/checkpoint selection."
