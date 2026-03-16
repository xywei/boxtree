from __future__ import annotations


from functools import lru_cache

import loopy as lp
import numpy as np

from boxtree.constant_one import ConstantOneDualTreeExpansionWrangler
from boxtree.dual_tree_traversal import _is_classic_list2_pair, _is_classic_list4_pair
from boxtree.tree import box_flags_enum


INTERACTION_SPLIT = np.int32(0)
INTERACTION_P2P = np.int32(1)
INTERACTION_M2L = np.int32(2)
INTERACTION_M2P = np.int32(3)
INTERACTION_P2L = np.int32(4)
INTERACTION_SKIP = np.int32(5)


@lru_cache(maxsize=None)
def _get_classify_frontier_knl(dim: int):
    center_dist = "\n".join(
        f"        <> d{i} = abs(box_centers[{i}, tgt_box] - box_centers[{i}, src_box])"
        for i in range(dim)
    )
    if dim == 1:
        max_dist = "d0"
    else:
        max_dist = f"fmax(d0, d1)"
        for i in range(2, dim):
            max_dist = f"fmax({max_dist}, d{i})"

    min_rad = "fmin(tgt_rad, src_rad)"

    knl = lp.make_kernel(
        "{[i]: 0<=i<n}",
        f"""
        for i
            <> src_box = frontier_src[i]
            <> tgt_box = frontier_tgt[i]
            <> src_flags = box_flags[src_box]
            <> tgt_flags = box_flags[tgt_box]
            <> src_nonempty = src_flags & {int(box_flags_enum.IS_SOURCE_BOX | box_flags_enum.IS_TARGET_BOX | box_flags_enum.HAS_SOURCE_CHILD_BOXES | box_flags_enum.HAS_TARGET_CHILD_BOXES)}
            <> tgt_has_targets = tgt_flags & {int(box_flags_enum.IS_TARGET_BOX | box_flags_enum.HAS_TARGET_CHILD_BOXES)}
            <> kind = {int(INTERACTION_SKIP)}
            if src_nonempty == 0 or tgt_has_targets == 0
                kind = {int(INTERACTION_SKIP)}
            else
                <> src_level = box_levels[src_box]
                <> tgt_level = box_levels[tgt_box]
                <> src_rad = root_extent * pow(0.5, src_level + 1)
                <> tgt_rad = root_extent * pow(0.5, tgt_level + 1)
{center_dist}
                <> linf_dist = {max_dist}
                <> adj_slack = (tgt_rad + src_rad) + {min_rad}
                <> ws_slack = ((2*(well_sep_is_n_away-1)+1)*tgt_rad + src_rad) + {min_rad}
                if linf_dist <= adj_slack
                    kind = {int(INTERACTION_SPLIT)}
                elif linf_dist <= ws_slack
                    kind = {int(INTERACTION_SPLIT)}
                else
                    if src_level == tgt_level
                        kind = {int(INTERACTION_M2L)}
                    elif (src_flags & {int(box_flags_enum.IS_SOURCE_BOX | box_flags_enum.HAS_SOURCE_CHILD_BOXES)}) == 0
                        kind = {int(INTERACTION_SKIP)}
                    elif src_level > tgt_level and (tgt_flags & {int(box_flags_enum.IS_TARGET_BOX)}) != 0
                        kind = {int(INTERACTION_M2P)}
                    elif (src_flags & {int(box_flags_enum.IS_SOURCE_BOX)}) != 0
                        kind = {int(INTERACTION_P2L)}
                    else
                        kind = {int(INTERACTION_SPLIT)}
                    end
                end
            end
            interaction[i] = kind
        end
        """,
        [
            lp.GlobalArg("frontier_src, frontier_tgt", np.int32, shape="n"),
            lp.GlobalArg("box_levels, box_flags", np.int32, shape="nboxes"),
            lp.GlobalArg("box_centers", np.float64, shape=(dim, "nboxes")),
            lp.GlobalArg("interaction", np.int32, shape="n"),
            lp.ValueArg("n, nboxes", np.int32),
            lp.ValueArg("root_extent", np.float64),
            lp.ValueArg("well_sep_is_n_away", np.int32),
        ],
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        name=f"classify_frontier_{dim}d",
    )
    return knl


def drive_frontier_constant_one_fmm(
    actx, wrangler: ConstantOneDualTreeExpansionWrangler, src_weight_vecs
):
    (src_weights,) = src_weight_vecs
    tree = wrangler.tree
    dim = tree.box_centers.shape[0]

    frontier_src = np.array([0], dtype=tree.box_id_dtype)
    frontier_tgt = np.array([0], dtype=tree.box_id_dtype)
    mpoles = wrangler.form_multipoles(
        actx,
        np.arange(tree.nlevels + 1, dtype=tree.box_id_dtype),
        np.flatnonzero(tree.box_flags & box_flags_enum.IS_SOURCE_BOX).astype(
            tree.box_id_dtype
        ),
        [wrangler.reorder_sources(src_weights)],
    )
    mpoles = wrangler.coarsen_multipoles(
        actx,
        np.arange(tree.nlevels + 1, dtype=tree.box_id_dtype),
        np.flatnonzero(tree.box_flags & box_flags_enum.HAS_SOURCE_CHILD_BOXES).astype(
            tree.box_id_dtype
        ),
        mpoles,
    )
    local_exps = wrangler.local_expansion_zeros(None)
    potentials = wrangler.output_zeros(None)

    knl = _get_classify_frontier_knl(dim).executor(actx.context)
    reordered_weights = [wrangler.reorder_sources(src_weights)]

    while len(frontier_src):
        interaction = np.empty(len(frontier_src), dtype=np.int32)
        _evt, (interaction_dev,) = knl(
            actx.queue,
            frontier_src=actx.from_numpy(frontier_src),
            frontier_tgt=actx.from_numpy(frontier_tgt),
            box_levels=actx.from_numpy(tree.box_levels),
            box_flags=actx.from_numpy(tree.box_flags),
            box_centers=actx.from_numpy(tree.box_centers),
            interaction=actx.from_numpy(interaction),
            n=np.int32(len(frontier_src)),
            nboxes=np.int32(tree.nboxes),
            root_extent=np.float64(tree.root_extent),
            well_sep_is_n_away=np.int32(wrangler.traversal_engine.well_sep_is_n_away),
        )
        interaction = actx.to_numpy(interaction_dev)

        next_src = []
        next_tgt = []
        for src_box, tgt_box, kind in zip(
            frontier_src, frontier_tgt, interaction, strict=True
        ):
            src_box = int(src_box)
            tgt_box = int(tgt_box)
            if kind == INTERACTION_SPLIT:
                src_leaf = not np.any(tree.box_child_ids[:, src_box])
                tgt_leaf = not np.any(tree.box_child_ids[:, tgt_box])
                src_level = int(tree.box_levels[src_box])
                tgt_level = int(tree.box_levels[tgt_box])
                if src_leaf:
                    children = [int(c) for c in tree.box_child_ids[:, tgt_box] if c]
                    next_src.extend([src_box] * len(children))
                    next_tgt.extend(children)
                elif tgt_leaf or src_level < tgt_level:
                    children = [int(c) for c in tree.box_child_ids[:, src_box] if c]
                    next_src.extend(children)
                    next_tgt.extend([tgt_box] * len(children))
                else:
                    children = [int(c) for c in tree.box_child_ids[:, tgt_box] if c]
                    next_src.extend([src_box] * len(children))
                    next_tgt.extend(children)
            else:
                pair_batch = type(
                    "PairBatch",
                    (),
                    {
                        "source_boxes": np.array([src_box], dtype=tree.box_id_dtype),
                        "target_boxes": np.array([tgt_box], dtype=tree.box_id_dtype),
                    },
                )()
                if kind == INTERACTION_P2P:
                    potentials = wrangler.eval_direct_batch(
                        actx, pair_batch, reordered_weights, potentials
                    )
                elif kind == INTERACTION_M2L:
                    local_exps = wrangler.multipole_to_local_batch(
                        actx, pair_batch, mpoles, local_exps
                    )
                elif kind == INTERACTION_M2P:
                    potentials = wrangler.eval_multipoles_batch(
                        actx, pair_batch, mpoles, potentials
                    )
                elif kind == INTERACTION_P2L:
                    local_exps = wrangler.form_locals_batch(
                        actx, pair_batch, reordered_weights, local_exps
                    )

        frontier_src = np.array(next_src, dtype=tree.box_id_dtype)
        frontier_tgt = np.array(next_tgt, dtype=tree.box_id_dtype)

    level_start_target = np.arange(tree.nlevels + 1, dtype=tree.box_id_dtype)
    target_boxes = np.flatnonzero(tree.box_flags & box_flags_enum.IS_TARGET_BOX).astype(
        tree.box_id_dtype
    )
    target_or_parent_boxes = np.flatnonzero(
        tree.box_flags
        & (box_flags_enum.IS_TARGET_BOX | box_flags_enum.HAS_TARGET_CHILD_BOXES)
    ).astype(tree.box_id_dtype)
    local_exps = wrangler.refine_locals(
        actx, level_start_target, target_or_parent_boxes, local_exps
    )
    potentials = potentials + wrangler.eval_locals(
        actx, level_start_target, target_boxes, local_exps
    )
    return wrangler.finalize_potentials(actx, wrangler.reorder_potentials(potentials))
