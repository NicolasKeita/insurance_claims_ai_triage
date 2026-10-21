"""Images."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import PurePosixPath

from vision.dataset._files import _check_image, _contained_file, _relative_filename


def _copy_coco_images(conversion, split_name, split, stage, rows):
    conversion.expected[split_name] = {}
    prepared_files = []
    for image in split.document["images"]:
        image_id = image["id"]
        filename = _relative_filename(image["file_name"])
        source = _contained_file(split.image_dir, filename)
        _check_image(source, image["width"], image["height"])
        image_path = stage / "images" / split_name / filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, image_path)
        label_relative = (PurePosixPath("labels") / split_name / PurePosixPath(filename).with_suffix(".txt")).as_posix()
        label_path = stage / label_relative
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "".join(
                str(class_id) + " " + " ".join(format(value, ".12g") for value in polygon) + "\n"
                for class_id, polygon in rows[image_id]
            ),
            encoding="utf-8",
        )
        conversion.expected[split_name][filename] = rows[image_id]
        prepared_files.append(
            {
                "image": (PurePosixPath("images") / split_name / filename).as_posix(),
                "label": label_relative,
                "image_id": image_id,
                "width": image["width"],
                "height": image["height"],
                "row_count": len(rows[image_id]),
                "class_counts": dict(Counter(str(row[0]) for row in rows[image_id])),
            }
        )
    return prepared_files
