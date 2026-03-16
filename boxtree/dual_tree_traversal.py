"""Prototype dual-tree traversal support for on-the-fly FMM interactions.

The current implementation emits batches of source-target box pairs classified
into the same broad interaction categories as the classic list-based FMM path.
It is intended as a correctness and API prototype.
"""

from __future__ import annotations


from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Literal

import numpy as np

from boxtree.tree import box_flags_enum


InteractionKind = Literal["p2p", "m2l", "m2p", "p2l"]


@dataclass(frozen=True)
class PairBatch:
    interaction_kind: InteractionKind
    source_boxes: np.ndarray
    target_boxes: np.ndarray


class DualTreeTraversalEngine:
    def __init__(self, well_sep_is_n_away: int = 1) -> None:
        if well_sep_is_n_away < 1:
            raise ValueError("well_sep_is_n_away must be at least 1")

        self.well_sep_is_n_away = well_sep_is_n_away

    def walk(self, tree):
        batches: dict[InteractionKind, list[tuple[int, int]]] = {
            "p2p": [],
            "m2l": [],
            "m2p": [],
            "p2l": [],
        }

        def add_pair(interaction_kind, source_box, target_box, _level=None):
            batches[interaction_kind].append((source_box, target_box))

        self.walk_streaming(tree, pair_visitor=add_pair)

        for interaction_kind, pairs in batches.items():
            if not pairs:
                continue

            source_boxes = np.array([src for src, _ in pairs], dtype=tree.box_id_dtype)
            target_boxes = np.array([tgt for _, tgt in pairs], dtype=tree.box_id_dtype)
            yield PairBatch(interaction_kind, source_boxes, target_boxes)

    def walk_streaming(self, tree, pair_visitor):
        seen_pairs: dict[InteractionKind, set[tuple[int, int]]] = {
            "p2p": set(),
            "m2l": set(),
            "m2p": set(),
            "p2l": set(),
        }

        pending_pairs: list[tuple[tuple[int, int, int, int], tuple[int, int]]] = []
        heappush(pending_pairs, (_pair_priority(tree, 0, 0), (0, 0)))

        while pending_pairs:
            _, (source_box, target_box) = heappop(pending_pairs)

            if not (
                _box_is_nonempty(tree, source_box)
                and _box_has_targets(tree, target_box)
            ):
                continue

            interaction = self._classify(tree, source_box, target_box)

            if interaction == "skip":
                continue

            if interaction != "split":
                pair = (source_box, target_box)
                if pair not in seen_pairs[interaction]:
                    seen_pairs[interaction].add(pair)
                    pair_visitor(
                        interaction,
                        source_box,
                        target_box,
                        int(tree.box_levels[target_box]),
                    )
                continue

            source_leaf = _is_leaf(tree, source_box)
            target_leaf = _is_leaf(tree, target_box)

            if source_leaf and target_leaf:
                if _box_has_direct_sources(tree, source_box):
                    pair = (source_box, target_box)
                    if pair not in seen_pairs["p2p"]:
                        seen_pairs["p2p"].add(pair)
                        pair_visitor(
                            "p2p",
                            source_box,
                            target_box,
                            int(tree.box_levels[target_box]),
                        )
                continue

            source_level = int(tree.box_levels[source_box])
            target_level = int(tree.box_levels[target_box])

            if source_leaf:
                for child in _target_children(tree, target_box):
                    heappush(
                        pending_pairs,
                        (_pair_priority(tree, source_box, child), (source_box, child)),
                    )
            elif target_leaf:
                for child in _interaction_source_children(tree, source_box):
                    heappush(
                        pending_pairs,
                        (_pair_priority(tree, child, target_box), (child, target_box)),
                    )
            elif source_level < target_level:
                for child in _interaction_source_children(tree, source_box):
                    heappush(
                        pending_pairs,
                        (_pair_priority(tree, child, target_box), (child, target_box)),
                    )
            elif target_level < source_level:
                for child in _target_children(tree, target_box):
                    heappush(
                        pending_pairs,
                        (_pair_priority(tree, source_box, child), (source_box, child)),
                    )
            else:
                for child in _target_children(tree, target_box):
                    heappush(
                        pending_pairs,
                        (_pair_priority(tree, source_box, child), (source_box, child)),
                    )

    def _classify(self, tree, source_box: int, target_box: int):
        return _classify_impl(self, tree, source_box, target_box)


def _pair_priority(tree, source_box: int, target_box: int) -> tuple[int, int, int, int]:
    return (
        int(tree.box_levels[target_box]),
        target_box,
        int(tree.box_levels[source_box]),
        source_box,
    )


def _classify_impl(self, tree, source_box: int, target_box: int):
    if _is_adjacent_or_overlapping(tree, source_box, target_box):
        return "split"

    source_level = int(tree.box_levels[source_box])
    target_level = int(tree.box_levels[target_box])

    if not _is_well_separated(tree, source_box, target_box, self.well_sep_is_n_away):
        return "split"

    if source_level == target_level and _is_classic_list2_pair(
        tree, source_box, target_box, self.well_sep_is_n_away
    ):
        return "m2l"

    if not _box_has_sources(tree, source_box):
        return "skip"

    if source_level > target_level and _box_has_direct_targets(tree, target_box):
        return "m2p"
    if _is_classic_list4_pair(tree, source_box, target_box, self.well_sep_is_n_away):
        return "p2l"
    return "split"


def _is_leaf(tree, box_id: int) -> bool:
    return not np.any(tree.box_child_ids[:, box_id])


def _box_has_sources(tree, box_id: int) -> bool:
    flags = int(tree.box_flags[box_id])
    return bool(
        flags & (box_flags_enum.IS_SOURCE_BOX | box_flags_enum.HAS_SOURCE_CHILD_BOXES)
    )


def _box_is_nonempty(tree, box_id: int) -> bool:
    flags = int(tree.box_flags[box_id])
    return bool(
        flags
        & (
            box_flags_enum.IS_SOURCE_BOX
            | box_flags_enum.IS_TARGET_BOX
            | box_flags_enum.HAS_SOURCE_CHILD_BOXES
            | box_flags_enum.HAS_TARGET_CHILD_BOXES
        )
    )


def _box_has_direct_sources(tree, box_id: int) -> bool:
    return bool(int(tree.box_flags[box_id]) & box_flags_enum.IS_SOURCE_BOX)


def _box_has_targets(tree, box_id: int) -> bool:
    flags = int(tree.box_flags[box_id])
    return bool(
        flags & (box_flags_enum.IS_TARGET_BOX | box_flags_enum.HAS_TARGET_CHILD_BOXES)
    )


def _box_has_direct_targets(tree, box_id: int) -> bool:
    return bool(int(tree.box_flags[box_id]) & box_flags_enum.IS_TARGET_BOX)


def _source_children(tree, box_id: int) -> list[int]:
    return [
        int(child)
        for child in tree.box_child_ids[:, box_id]
        if child and _box_has_sources(tree, int(child))
    ]


def _interaction_source_children(tree, box_id: int) -> list[int]:
    return [
        int(child)
        for child in tree.box_child_ids[:, box_id]
        if child and _box_is_nonempty(tree, int(child))
    ]


def _target_children(tree, box_id: int) -> list[int]:
    return [
        int(child)
        for child in tree.box_child_ids[:, box_id]
        if child and _box_has_targets(tree, int(child))
    ]


def _level_to_rad(tree, level: int):
    return tree.root_extent * (0.5 ** (level + 1))


def _linf_dist(tree, source_box: int, target_box: int):
    return np.max(
        np.abs(tree.box_centers[:, target_box] - tree.box_centers[:, source_box])
    )


def _is_adjacent_or_overlapping_with_neighborhood(
    tree, source_box: int, target_box: int, target_box_neighborhood_size: int
) -> bool:
    target_level = int(tree.box_levels[target_box])
    source_level = int(tree.box_levels[source_box])
    target_rad = _level_to_rad(tree, target_level)
    source_rad = _level_to_rad(tree, source_level)
    rad_sum = (2 * (target_box_neighborhood_size - 1) + 1) * target_rad + source_rad
    slack = rad_sum + min(target_rad, source_rad)
    return _linf_dist(tree, source_box, target_box) <= slack


def _is_adjacent_or_overlapping(tree, source_box: int, target_box: int) -> bool:
    return _is_adjacent_or_overlapping_with_neighborhood(
        tree, source_box, target_box, 1
    )


def _is_well_separated(
    tree, source_box: int, target_box: int, well_sep_is_n_away: int
) -> bool:
    return not _is_adjacent_or_overlapping_with_neighborhood(
        tree, source_box, target_box, well_sep_is_n_away
    )


def _is_classic_list2_pair(
    tree, source_box: int, target_box: int, well_sep_is_n_away: int
) -> bool:
    source_level = int(tree.box_levels[source_box])
    target_level = int(tree.box_levels[target_box])

    if source_level != target_level or source_level == 0:
        return False

    source_parent = int(tree.box_parent_ids[source_box])
    target_parent = int(tree.box_parent_ids[target_box])

    if source_parent < 0 or target_parent < 0:
        return False

    if source_parent == target_parent:
        return False

    if source_parent not in _same_level_non_well_sep_boxes(
        tree, target_parent, well_sep_is_n_away
    ):
        return False

    return _is_well_separated(tree, source_box, target_box, well_sep_is_n_away)


def _is_classic_list4_pair(
    tree, source_box: int, target_box: int, well_sep_is_n_away: int
) -> bool:
    if not _box_has_direct_sources(tree, source_box):
        return False

    source_level = int(tree.box_levels[source_box])
    target_level = int(tree.box_levels[target_box])

    if source_level >= target_level or target_level == 0:
        return False

    target_parent = int(tree.box_parent_ids[target_box])
    if target_parent < 0:
        return False

    if not _is_well_separated(tree, source_box, target_box, well_sep_is_n_away):
        return False

    if (
        tree.sources_have_extent or tree.targets_have_extent
    ) and not _meets_sep_bigger_criterion(tree, target_box, source_box):
        return False

    immediate_parent_level = target_level - 1

    if well_sep_is_n_away == 1:
        walk_level = target_level - 1
        current_target_parent = target_parent
    else:
        walk_level = target_level
        current_target_parent = target_box

    while walk_level >= 0:
        if source_level == walk_level and source_box in _same_level_non_well_sep_boxes(
            tree, current_target_parent, well_sep_is_n_away
        ):
            if walk_level < target_level:
                if _is_adjacent_or_overlapping(tree, source_box, target_parent):
                    return True

                if tree.sources_have_extent or tree.targets_have_extent:
                    return not _meets_sep_bigger_criterion(
                        tree, target_parent, source_box
                    )

                return False
            return True

        if walk_level == 0:
            break

        current_target_parent = int(tree.box_parent_ids[current_target_parent])
        walk_level -= 1

    return False


def _meets_sep_bigger_criterion(tree, target_box: int, source_box: int) -> bool:
    target_level = int(tree.box_levels[target_box])
    source_level = int(tree.box_levels[source_box])
    target_rad = _level_to_rad(tree, target_level)
    source_rad = _level_to_rad(tree, source_level)
    max_allowed_center_l_inf_dist = (
        3 * (1 + tree.stick_out_factor) * target_rad + source_rad
    )
    l_inf_dist = _linf_dist(tree, source_box, target_box)
    return l_inf_dist >= max_allowed_center_l_inf_dist * (
        1 - 8 * np.finfo(tree.coord_dtype).eps
    )


def _same_level_non_well_sep_boxes(
    tree, box_id: int, well_sep_is_n_away: int
) -> tuple[int, ...]:
    if box_id == 0:
        return ()

    level = int(tree.box_levels[box_id])
    result: list[int] = []
    stack = [0]

    while stack:
        walk_box_id = stack.pop()
        if walk_box_id == 0:
            child_ids = tree.box_child_ids[:, 0]
        else:
            child_ids = tree.box_child_ids[:, walk_box_id]

        for child in child_ids:
            if child == 0:
                continue

            child = int(child)
            if not _is_adjacent_or_overlapping_with_neighborhood(
                tree, box_id, child, well_sep_is_n_away
            ):
                continue

            child_level = int(tree.box_levels[child])
            if child_level == level:
                if child != box_id:
                    result.append(child)
            elif child_level < level:
                stack.append(child)

    return tuple(result)
