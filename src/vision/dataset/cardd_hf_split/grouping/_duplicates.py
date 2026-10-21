from __future__ import annotations

from collections import Counter, defaultdict

from vision.dataset.cardd_hf_split._identity import _Group


def _duplicate_groups(samples, paths, hashes, sample_labels, classes):
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for indices in (*paths.values(), *hashes.values()):
        for index in indices[1:]:
            union(indices[0], index)
    members = defaultdict(list)
    for index, sample in enumerate(samples):
        members[find(index)].append(sample.sample_id)
    groups = []
    for ids in sorted(members.values()):
        counts = Counter(label for sample_id in ids for label in sample_labels[sample_id])
        groups.append(_Group(tuple(ids), {label: counts[label] for label in classes if counts[label]}))
    return groups


def _duplicate_metadata(groups, paths, hashes, names):
    return {
        "policy": "same_resolved_filepath_or_identical_image_sha256_together",
        "group_count": len(groups),
        "duplicate_group_count": sum(group.size > 1 for group in groups),
        "samples_in_duplicate_groups": sum(group.size for group in groups if group.size > 1),
        "duplicate_filepath_groups": sum(len(indices) > 1 for indices in paths.values()),
        "duplicate_image_sha256_groups": sum(len(indices) > 1 for indices in hashes.values()),
        "duplicate_filename_groups": sum(len(indices) > 1 for indices in names.values()),
        "groups": [list(group.ids) for group in groups if group.size > 1],
    }
