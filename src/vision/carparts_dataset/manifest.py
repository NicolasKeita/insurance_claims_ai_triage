"""Reproducibility manifest built from a real inspection report."""

from vision.carparts_dataset.constants import CONFIG_PATH, METRICS_SCOPE, OFFICIAL_ARCHIVE, OFFICIAL_SOURCE
from vision.training import file_sha256


def build_manifest(inspection, paths):
    import ultralytics

    dataset_root = paths["dataset_root"]
    license_files = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and ("license" in path.name.lower() or "readme" in path.name.lower())
    )
    return inspection | {
        "schema_version": 1,
        "source": OFFICIAL_SOURCE,
        "source_name": "Ultralytics official Carparts-Seg",
        "source_repository": OFFICIAL_ARCHIVE,
        "license": {
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "evidence": "CC BY 4.0 badge on Ultralytics dataset documentation",
            "attribution": "Gianmarco Russo; car-seg dataset on Roboflow Universe (2023)",
            "archive_metadata_files": [path.relative_to(dataset_root).as_posix() for path in license_files],
        },
        "ultralytics_version": str(ultralytics.__version__),
        "source_dataset_yaml_sha256": file_sha256(paths["copied_source"]),
        "installed_official_yaml_sha256": file_sha256(paths["official_yaml"]),
        "relevant_files": {
            "source_yaml": str(paths["copied_source"]),
            "training_yaml": str(paths["dataset_yaml"]),
            "installed_official_yaml": str(paths["official_yaml"]),
            "training_config": CONFIG_PATH,
        },
        "metrics_scope": METRICS_SCOPE,
    }
