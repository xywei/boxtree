from __future__ import annotations


import importlib
import importlib.util
from functools import partial

import numpy as np
import pytest

from arraycontext import pytest_generate_tests_for_array_contexts
from pytools import obj_array

from boxtree.array_context import (
    PytestPyOpenCLArrayContextFactory,
    _acf,  # noqa: F401
)
from boxtree.constant_one import (
    ConstantOneDualTreeExpansionWrangler,
    ConstantOneExpansionWrangler,
    ConstantOneTreeIndependentDataForWrangler,
)
from boxtree.dual_tree_fmm import (
    CurrentLevelCSRAccumulator,
    CSRBatchAdapter,
    DualTreeCompatibilityWrangler,
    LevelwiseM2LSchedule,
    M2LLevelSchedule,
    LevelwiseCSRAccumulator,
    drive_dual_tree_fmm,
)
from boxtree.frontier_fmm import drive_frontier_constant_one_fmm
from boxtree.dual_tree_traversal import DualTreeTraversalEngine
from boxtree.fmm import drive_fmm
from boxtree.tree_of_boxes import make_tree_of_boxes_root, refine_tree_of_boxes
from boxtree.tools import make_normal_particle_array


pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
])


def _import_local_sumpy():
    if importlib.util.find_spec("sumpy") is None:
        pytest.skip("sumpy not importable in this environment")

    metadata = importlib.import_module("importlib.metadata")
    orig_version = metadata.version

    try:
        metadata.version = lambda name: "0" if name == "sumpy" else orig_version(name)
        return {
            "PyOpenCLArrayContext": importlib.import_module(
                "sumpy.array_context"
            ).PyOpenCLArrayContext,
            "NonFFTM2LTranslationClassFactory": importlib.import_module(
                "sumpy.expansion.m2l"
            ).NonFFTM2LTranslationClassFactory,
            "VolumeTaylorLocalExpansion": importlib.import_module(
                "sumpy.expansion.local"
            ).VolumeTaylorLocalExpansion,
            "VolumeTaylorMultipoleExpansion": importlib.import_module(
                "sumpy.expansion.multipole"
            ).VolumeTaylorMultipoleExpansion,
            "SumpyExpansionWrangler": importlib.import_module(
                "sumpy.fmm"
            ).SumpyExpansionWrangler,
            "SumpyTreeIndependentDataForWrangler": importlib.import_module(
                "sumpy.fmm"
            ).SumpyTreeIndependentDataForWrangler,
            "LaplaceKernel": importlib.import_module("sumpy.kernel").LaplaceKernel,
            "HelmholtzKernel": importlib.import_module("sumpy.kernel").HelmholtzKernel,
        }
    finally:
        metadata.version = orig_version


def _run_sumpy_dual_tree_case(
    actx,
    *,
    knl,
    local_expn_class,
    mpole_expn_class,
    sources,
    targets,
    max_particles_in_box,
    order,
    dtype,
    weight_seed,
    kernel_extra_kwargs=None,
    m2l_translation_factory=None,
):
    classic, compat, weights = _build_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=local_expn_class,
        mpole_expn_class=mpole_expn_class,
        sources=sources,
        targets=targets,
        max_particles_in_box=max_particles_in_box,
        order=order,
        dtype=dtype,
        weight_seed=weight_seed,
        kernel_extra_kwargs=kernel_extra_kwargs,
        m2l_translation_factory=m2l_translation_factory,
    )

    (classic_result,) = drive_fmm(actx, classic, (weights,))
    (compat_result,) = drive_dual_tree_fmm(actx, compat, (weights,))
    return actx.to_numpy(classic_result), actx.to_numpy(compat_result)


def _build_sumpy_dual_tree_case(
    actx,
    *,
    knl,
    local_expn_class,
    mpole_expn_class,
    sources,
    targets,
    max_particles_in_box,
    order,
    dtype,
    weight_seed,
    kernel_extra_kwargs=None,
    m2l_translation_factory=None,
):
    from boxtree import TreeBuilder
    from boxtree.traversal import FMMTraversalBuilder

    if kernel_extra_kwargs is None:
        kernel_extra_kwargs = {}

    tb = TreeBuilder(actx)
    tree, _ = tb(
        actx,
        sources,
        targets=targets,
        max_particles_in_box=max_particles_in_box,
        debug=True,
    )

    tbuild = FMMTraversalBuilder(actx, well_sep_is_n_away=1)
    trav, _ = tbuild(actx, tree, debug=True)

    local_factory = partial(local_expn_class, knl)
    if m2l_translation_factory is not None:
        m2l_translation = m2l_translation_factory.get_m2l_translation_class(
            knl, local_expn_class
        )()
        local_factory = partial(
            local_expn_class, knl, m2l_translation_override=m2l_translation
        )

    sumpy = _import_local_sumpy()
    SumpyExpansionWrangler = sumpy["SumpyExpansionWrangler"]
    SumpyTreeIndependentDataForWrangler = sumpy["SumpyTreeIndependentDataForWrangler"]

    tree_indep = SumpyTreeIndependentDataForWrangler(
        actx,
        partial(mpole_expn_class, knl),
        local_factory,
        [knl],
    )

    def fmm_level_to_order(kernel, kernel_args, tree, lev):
        del kernel, kernel_args, tree, lev
        return order

    classic = SumpyExpansionWrangler(
        tree_indep,
        trav,
        dtype,
        fmm_level_to_order=fmm_level_to_order,
        kernel_extra_kwargs=kernel_extra_kwargs,
        _disable_translation_classes=(m2l_translation_factory is not None),
    )
    compat = DualTreeCompatibilityWrangler(
        tree_indep,
        actx.to_numpy(tree),
        DualTreeTraversalEngine(well_sep_is_n_away=1),
        classic,
    )

    rng = np.random.default_rng(weight_seed)
    weights = actx.from_numpy(rng.random(len(sources[0]), dtype=np.float64)).astype(
        dtype
    )
    return classic, compat, weights


def _assert_allclose_maybe_obj_array(actual, expected):
    if isinstance(actual, np.ndarray) and actual.dtype == object:
        assert isinstance(expected, np.ndarray) and expected.dtype == object
        assert len(actual) == len(expected)
        for act_item, exp_item in zip(actual, expected, strict=True):
            assert np.allclose(act_item, exp_item)
    else:
        assert np.allclose(actual, expected)


def _run_sumpy_dual_tree_intermediate_case(
    actx,
    *,
    knl,
    local_expn_class,
    mpole_expn_class,
    sources,
    targets,
    max_particles_in_box,
    order,
    dtype,
    weight_seed,
    kernel_extra_kwargs=None,
    m2l_translation_factory=None,
):
    classic, compat, weights = _build_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=local_expn_class,
        mpole_expn_class=mpole_expn_class,
        sources=sources,
        targets=targets,
        max_particles_in_box=max_particles_in_box,
        order=order,
        dtype=dtype,
        weight_seed=weight_seed,
        kernel_extra_kwargs=kernel_extra_kwargs,
        m2l_translation_factory=m2l_translation_factory,
    )

    src_weight_vecs = (weights,)
    traversal = classic.traversal

    reordered_weights = [classic.reorder_sources(weight) for weight in src_weight_vecs]

    classic_mpoles = classic.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        reordered_weights,
    )
    classic_mpoles = classic.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        classic_mpoles,
    )

    classic_non_qbx = classic.eval_direct(
        actx,
        traversal.target_boxes,
        traversal.neighbor_source_boxes_starts,
        traversal.neighbor_source_boxes_lists,
        reordered_weights,
    )
    classic_locals = classic.multipole_to_local(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_siblings_starts,
        traversal.from_sep_siblings_lists,
        classic_mpoles,
    )
    classic_non_qbx = classic_non_qbx + classic.eval_multipoles(
        actx,
        traversal.target_boxes_sep_smaller_by_source_level,
        traversal.from_sep_smaller_by_level,
        classic_mpoles,
    )
    classic_locals = classic_locals + classic.form_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        traversal.from_sep_bigger_starts,
        traversal.from_sep_bigger_lists,
        reordered_weights,
    )
    classic_locals = classic.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        classic_locals,
    )
    classic_eval_locals = classic.eval_locals(
        actx,
        traversal.level_start_target_box_nrs,
        traversal.target_boxes,
        classic_locals,
    )

    dual_mpoles = compat.form_multipoles(
        actx,
        traversal.level_start_source_box_nrs,
        traversal.source_boxes,
        reordered_weights,
    )
    dual_mpoles = compat.coarsen_multipoles(
        actx,
        traversal.level_start_source_parent_box_nrs,
        traversal.source_parent_boxes,
        dual_mpoles,
    )
    dual_locals = compat.local_expansion_zeros(actx)
    dual_non_qbx = compat.output_zeros(actx)
    for pair_batch in compat.traversal_engine.walk(compat.tree):
        if pair_batch.interaction_kind == "p2p":
            dual_non_qbx = compat.eval_direct_batch(
                actx, pair_batch, reordered_weights, dual_non_qbx
            )
        elif pair_batch.interaction_kind == "m2l":
            dual_locals = compat.multipole_to_local_batch(
                actx, pair_batch, dual_mpoles, dual_locals
            )
        elif pair_batch.interaction_kind == "m2p":
            dual_non_qbx = compat.eval_multipoles_batch(
                actx, pair_batch, dual_mpoles, dual_non_qbx
            )
        elif pair_batch.interaction_kind == "p2l":
            dual_locals = compat.form_locals_batch(
                actx, pair_batch, reordered_weights, dual_locals
            )

    dual_locals = compat.refine_locals(
        actx,
        traversal.level_start_target_or_target_parent_box_nrs,
        traversal.target_or_target_parent_boxes,
        dual_locals,
    )
    dual_eval_locals = compat.eval_locals(
        actx,
        traversal.level_start_target_box_nrs,
        traversal.target_boxes,
        dual_locals,
    )

    return {
        "classic_mpoles": actx.to_numpy(classic_mpoles),
        "dual_mpoles": actx.to_numpy(dual_mpoles),
        "classic_non_qbx": actx.to_numpy(classic_non_qbx),
        "dual_non_qbx": actx.to_numpy(dual_non_qbx),
        "classic_locals": actx.to_numpy(classic_locals),
        "dual_locals": actx.to_numpy(dual_locals),
        "classic_eval_locals": actx.to_numpy(classic_eval_locals),
        "dual_eval_locals": actx.to_numpy(dual_eval_locals),
    }


@pytest.mark.opencl
@pytest.mark.parametrize(
    ("dims", "sources_are_targets"),
    [
        (2, True),
        (2, False),
        (3, True),
    ],
)
def test_dual_tree_fmm_constant_one_counts_all_sources(
    actx_factory, dims, sources_are_targets
):
    actx = actx_factory()

    nsources = 250
    sources = make_normal_particle_array(actx, nsources, dims, np.float64, seed=17)
    if sources_are_targets:
        targets = None
    else:
        targets = make_normal_particle_array(actx, 180, dims, np.float64, seed=21)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=12, debug=True)

    host_tree = actx.to_numpy(tree)
    tree_indep = ConstantOneTreeIndependentDataForWrangler()
    wrangler = ConstantOneDualTreeExpansionWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
    )

    weights = np.ones(nsources)
    result = drive_dual_tree_fmm(actx, wrangler, [weights])

    expected = np.full(host_tree.ntargets, nsources, dtype=np.float64)
    assert np.array_equal(result, expected)


@pytest.mark.opencl
def test_dual_tree_fmm_matches_classic_constant_one_fmm(actx_factory):
    actx = actx_factory()

    nsources = 220
    ntargets = 160
    sources = make_normal_particle_array(actx, nsources, 2, np.float64, seed=12)
    targets = make_normal_particle_array(actx, ntargets, 2, np.float64, seed=13)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=10, debug=True)

    from boxtree.traversal import FMMTraversalBuilder

    tbuild = FMMTraversalBuilder(actx, well_sep_is_n_away=1)
    trav, _ = tbuild(actx, tree, debug=True)

    host_trav = actx.to_numpy(trav)
    host_tree = host_trav.tree
    tree_indep = ConstantOneTreeIndependentDataForWrangler()

    classic = ConstantOneExpansionWrangler(tree_indep, host_trav)
    dual = ConstantOneDualTreeExpansionWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
    )

    weights = np.ones(nsources)
    classic_result = drive_fmm(actx, classic, [weights])
    dual_result = drive_dual_tree_fmm(actx, dual, [weights])

    assert np.array_equal(dual_result, classic_result)


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_matches_classic_constant_one(actx_factory):
    actx = actx_factory()

    nsources = 180
    ntargets = 130
    sources = make_normal_particle_array(actx, nsources, 2, np.float64, seed=31)
    targets = make_normal_particle_array(actx, ntargets, 2, np.float64, seed=32)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=10, debug=True)

    from boxtree.traversal import FMMTraversalBuilder

    tbuild = FMMTraversalBuilder(actx, well_sep_is_n_away=1)
    trav, _ = tbuild(actx, tree, debug=True)

    host_trav = actx.to_numpy(trav)
    host_tree = host_trav.tree
    tree_indep = ConstantOneTreeIndependentDataForWrangler()
    classic = ConstantOneExpansionWrangler(tree_indep, host_trav)
    compat = DualTreeCompatibilityWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
        classic,
    )

    weights = np.ones(nsources)
    classic_result = drive_fmm(actx, classic, [weights])
    compat_result = drive_dual_tree_fmm(actx, compat, [weights])

    assert np.array_equal(compat_result, classic_result)


def test_dual_tree_traversal_emits_all_interaction_kinds():
    tob = make_tree_of_boxes_root((np.array([0.0, 0.0]), np.array([1.0, 1.0])))

    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[0] = True
    tob = refine_tree_of_boxes(tob, refine_flags)

    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[1] = True
    tob = refine_tree_of_boxes(tob, refine_flags)

    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[4] = True
    tob = refine_tree_of_boxes(tob, refine_flags)

    refine_flags = np.zeros(tob.nboxes, dtype=bool)
    refine_flags[5] = True
    tob = refine_tree_of_boxes(tob, refine_flags)

    batches = list(DualTreeTraversalEngine(well_sep_is_n_away=1).walk(tob))
    interaction_kinds = {batch.interaction_kind for batch in batches}

    assert interaction_kinds == {"p2p", "m2l", "m2p", "p2l"}


def test_csr_batch_adapter_groups_pairs_by_target_box_dtype():
    pair_batch = type(
        "DummyPairBatch",
        (),
        {
            "source_boxes": np.array([7, 8, 2, 5, 3], dtype=np.int32),
            "target_boxes": np.array([4, 4, 1, 4, 1], dtype=np.int32),
        },
    )()

    csr_batch = CSRBatchAdapter(np.dtype(np.int32)).pair_batch_to_csr(pair_batch)

    assert np.array_equal(csr_batch.target_boxes, np.array([1, 4], dtype=np.int32))
    assert np.array_equal(csr_batch.starts, np.array([0, 2, 5], dtype=np.int32))
    assert np.array_equal(csr_batch.lists, np.array([2, 3, 5, 7, 8], dtype=np.int32))


def test_levelwise_csr_accumulator_builds_sorted_schedule():
    acc = LevelwiseCSRAccumulator(np.dtype(np.int32))
    acc.add_pairs(
        3, np.array([8, 5, 7], dtype=np.int32), np.array([4, 4, 1], dtype=np.int32)
    )
    acc.add_pairs(1, np.array([9, 2], dtype=np.int32), np.array([3, 3], dtype=np.int32))

    schedule = acc.as_schedule()

    assert [batch.level for batch in schedule.level_batches] == [1, 3]
    first, second = schedule.level_batches
    assert np.array_equal(first.csr_batch.target_boxes, np.array([3], dtype=np.int32))
    assert np.array_equal(first.csr_batch.starts, np.array([0, 2], dtype=np.int32))
    assert np.array_equal(first.csr_batch.lists, np.array([2, 9], dtype=np.int32))
    assert np.array_equal(
        second.csr_batch.target_boxes, np.array([1, 4], dtype=np.int32)
    )
    assert np.array_equal(second.csr_batch.starts, np.array([0, 1, 3], dtype=np.int32))
    assert np.array_equal(second.csr_batch.lists, np.array([7, 5, 8], dtype=np.int32))


def test_constant_one_native_levelwise_m2l_executor():
    schedule = LevelwiseM2LSchedule(
        level_batches=(
            M2LLevelSchedule(
                level=2,
                csr_batch=type(
                    "Batch",
                    (),
                    {
                        "target_boxes": np.array([3, 5], dtype=np.int32),
                        "starts": np.array([0, 2, 3], dtype=np.int32),
                        "lists": np.array([1, 4, 2], dtype=np.int32),
                    },
                )(),
            ),
        )
    )

    wrangler = ConstantOneDualTreeExpansionWrangler(
        ConstantOneTreeIndependentDataForWrangler(),
        type("DummyTree", (), {"box_id_dtype": np.int32, "nboxes": 8})(),
        None,
    )
    mpoles = np.array([0, 10, 20, 0, 40, 0, 0, 0], dtype=np.float64)
    locals_ = np.zeros(8, dtype=np.float64)

    result = wrangler.execute_levelwise_m2l_schedule(None, schedule, mpoles, locals_)

    assert np.array_equal(
        result, np.array([0, 0, 0, 50, 0, 20, 0, 0], dtype=np.float64)
    )


def test_current_level_csr_accumulator_builds_single_level_batch():
    acc = CurrentLevelCSRAccumulator(np.dtype(np.int32))
    acc.add_pair(4, 8, 5)
    acc.add_pair(4, 2, 5)
    acc.add_pair(4, 7, 3)

    batch = acc.to_level_batch()

    assert batch.level == 4
    assert np.array_equal(
        batch.csr_batch.target_boxes, np.array([3, 5], dtype=np.int32)
    )
    assert np.array_equal(batch.csr_batch.starts, np.array([0, 1, 3], dtype=np.int32))
    assert np.array_equal(batch.csr_batch.lists, np.array([7, 2, 8], dtype=np.int32))


@pytest.mark.opencl
def test_dual_tree_driver_uses_levelwise_m2l_executor(actx_factory):
    actx = actx_factory()

    nsources = 120
    ntargets = 90
    sources = make_normal_particle_array(actx, nsources, 2, np.float64, seed=121)
    targets = make_normal_particle_array(actx, ntargets, 2, np.float64, seed=122)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=8, debug=True)

    host_tree = actx.to_numpy(tree)
    tree_indep = ConstantOneTreeIndependentDataForWrangler()

    wrangler = ConstantOneDualTreeExpansionWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
    )

    weights = np.ones(nsources)
    result = drive_dual_tree_fmm(actx, wrangler, [weights])

    expected = np.full(host_tree.ntargets, nsources, dtype=np.float64)
    assert np.array_equal(result, expected)
    assert wrangler.uses_native_levelwise_m2l()
    assert wrangler.uses_traversal_free_m2l()
    stats = wrangler.get_execution_stats()
    assert stats is not None
    assert not stats.used_levelwise_m2l_schedule


@pytest.mark.opencl
@pytest.mark.xfail(reason="Loopy frontier-evolution prototype is not complete yet")
def test_frontier_constant_one_matches_dual_tree_driver(actx_factory):
    actx = actx_factory()

    nsources = 80
    ntargets = 70
    sources = make_normal_particle_array(actx, nsources, 2, np.float64, seed=131)
    targets = make_normal_particle_array(actx, ntargets, 2, np.float64, seed=132)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=8, debug=True)

    host_tree = actx.to_numpy(tree)
    tree_indep = ConstantOneTreeIndependentDataForWrangler()
    wrangler = ConstantOneDualTreeExpansionWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
    )

    weights = np.ones(nsources)
    dual_result = drive_dual_tree_fmm(actx, wrangler, [weights])
    frontier_result = drive_frontier_constant_one_fmm(actx, wrangler, [weights])

    assert np.array_equal(frontier_result, dual_result)


@pytest.mark.opencl
def test_dual_tree_pair_counts_match_classic_interactions(actx_factory):
    actx = actx_factory()

    sources = make_normal_particle_array(actx, 160, 2, np.float64, seed=41)
    targets = make_normal_particle_array(actx, 120, 2, np.float64, seed=42)

    from boxtree import TreeBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=10, debug=True)

    from boxtree.traversal import FMMTraversalBuilder

    tbuild = FMMTraversalBuilder(actx, well_sep_is_n_away=1)
    trav, _ = tbuild(actx, tree, debug=True)

    host_trav = actx.to_numpy(trav)
    host_tree = host_trav.tree
    pair_counts = {
        "p2p": 0,
        "m2l": 0,
        "m2p": 0,
        "p2l": 0,
    }
    for batch in DualTreeTraversalEngine(well_sep_is_n_away=1).walk(host_tree):
        if batch.interaction_kind == "m2l":
            pair_counts["m2l"] += sum(
                1
                for src_ibox in batch.source_boxes
                if host_tree.box_flags[src_ibox] & host_tree.box_flags.dtype.type(1)
            )
        else:
            pair_counts[batch.interaction_kind] += len(batch.source_boxes)

    nonempty_classic_m2l = 0
    for itgt_box, _tgt_ibox in enumerate(host_trav.target_or_target_parent_boxes):
        start, end = host_trav.from_sep_siblings_starts[itgt_box : itgt_box + 2]
        for src_ibox in host_trav.from_sep_siblings_lists[start:end]:
            if host_tree.box_flags[src_ibox] & host_tree.box_flags.dtype.type(1):
                nonempty_classic_m2l += 1

    classic_counts = {
        "p2p": int(host_trav.neighbor_source_boxes_starts[-1]),
        "m2l": nonempty_classic_m2l,
        "m2p": sum(
            int(level.starts[-1]) for level in host_trav.from_sep_smaller_by_level
        ),
        "p2l": int(host_trav.from_sep_bigger_starts[-1]),
    }

    assert pair_counts == classic_counts


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_matches_sumpy_direct_path(actx_factory):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    nsources = 40
    ntargets = 35
    sources = make_normal_particle_array(actx, nsources, 2, np.float64, seed=51)
    targets = make_normal_particle_array(actx, ntargets, 2, np.float64, seed=52)

    classic_result, compat_result = _run_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=max(nsources, ntargets) + 1,
        order=2,
        dtype=np.float64,
        weight_seed=53,
    )

    assert np.allclose(
        compat_result,
        classic_result,
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_matches_sumpy_far_field(actx_factory):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    rng = np.random.default_rng(61)
    nsources = 80
    ntargets = 1
    sources = obj_array.new_1d([
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
    ])
    targets = obj_array.new_1d([
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
    ])

    classic_result, compat_result = _run_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=4,
        order=3,
        dtype=np.float64,
        weight_seed=63,
        m2l_translation_factory=NonFFTM2LTranslationClassFactory(),
    )

    assert np.allclose(
        compat_result,
        classic_result,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_matches_sumpy_observable_stages(
    actx_factory,
):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    rng = np.random.default_rng(81)
    nsources = 60
    targets = obj_array.new_1d([
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
    ])
    sources = obj_array.new_1d([
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
    ])

    stage_data = _run_sumpy_dual_tree_intermediate_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=4,
        order=3,
        dtype=np.float64,
        weight_seed=82,
        m2l_translation_factory=NonFFTM2LTranslationClassFactory(),
    )

    _assert_allclose_maybe_obj_array(
        stage_data["dual_mpoles"], stage_data["classic_mpoles"]
    )
    _assert_allclose_maybe_obj_array(
        stage_data["dual_non_qbx"], stage_data["classic_non_qbx"]
    )
    _assert_allclose_maybe_obj_array(
        stage_data["dual_eval_locals"], stage_data["classic_eval_locals"]
    )


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_collects_m2l_stats(actx_factory):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    rng = np.random.default_rng(91)
    nsources = 60
    targets = obj_array.new_1d([
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
    ])
    sources = obj_array.new_1d([
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
    ])

    classic, compat, weights = _build_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=4,
        order=3,
        dtype=np.float64,
        weight_seed=92,
        m2l_translation_factory=NonFFTM2LTranslationClassFactory(),
    )
    del classic

    compat.reset_execution_stats()
    drive_dual_tree_fmm(actx, compat, (weights,))
    stats = compat.get_execution_stats()

    assert stats is not None
    assert stats.traversal_seconds >= 0
    assert stats.m2l_finalize_seconds >= 0
    assert stats.m2l_pair_count > 0
    assert stats.m2l_level_count > 0
    assert stats.used_levelwise_m2l_schedule


@pytest.mark.opencl
def test_dual_tree_compatibility_wrangler_uses_buffered_m2l_executor(actx_factory):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    rng = np.random.default_rng(95)
    nsources = 80
    targets = obj_array.new_1d([
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
        actx.from_numpy(np.array([1.0], dtype=np.float64)),
    ])
    sources = obj_array.new_1d([
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
        actx.from_numpy(
            (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
        ),
    ])

    _classic, compat, weights = _build_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=4,
        order=3,
        dtype=np.float64,
        weight_seed=96,
        m2l_translation_factory=NonFFTM2LTranslationClassFactory(),
    )

    compat.reset_execution_stats()
    drive_dual_tree_fmm(actx, compat, (weights,))
    stats = compat.get_execution_stats()

    assert stats is not None
    assert stats.used_levelwise_m2l_schedule
    assert stats.m2l_level_count >= 1


@pytest.mark.opencl
@pytest.mark.parametrize(
    ("dims", "nsources", "ntargets", "max_particles_in_box", "seed"),
    [
        (2, 80, 70, 6, 101),
        (2, 120, 90, 10, 102),
        (3, 90, 60, 8, 103),
        (3, 140, 110, 12, 104),
    ],
)
def test_dual_tree_constant_one_matches_classic_randomized(
    actx_factory, dims, nsources, ntargets, max_particles_in_box, seed
):
    actx = actx_factory()

    sources = make_normal_particle_array(actx, nsources, dims, np.float64, seed=seed)
    targets = make_normal_particle_array(
        actx, ntargets, dims, np.float64, seed=seed + 1000
    )

    from boxtree import TreeBuilder
    from boxtree.traversal import FMMTraversalBuilder

    tb = TreeBuilder(actx)
    tree, _ = tb(
        actx,
        sources,
        targets=targets,
        max_particles_in_box=max_particles_in_box,
        debug=True,
    )
    tbuild = FMMTraversalBuilder(actx, well_sep_is_n_away=1)
    trav, _ = tbuild(actx, tree, debug=True)

    host_trav = actx.to_numpy(trav)
    host_tree = host_trav.tree
    tree_indep = ConstantOneTreeIndependentDataForWrangler()
    classic = ConstantOneExpansionWrangler(tree_indep, host_trav)
    dual = ConstantOneDualTreeExpansionWrangler(
        tree_indep,
        host_tree,
        DualTreeTraversalEngine(well_sep_is_n_away=1),
    )

    weights = np.ones(nsources)
    classic_result = drive_fmm(actx, classic, [weights])
    dual_result = drive_dual_tree_fmm(actx, dual, [weights])

    assert np.array_equal(dual_result, classic_result)


@pytest.mark.opencl
@pytest.mark.parametrize("seed", [111, 112, 113])
def test_dual_tree_sumpy_far_field_randomized(seed, actx_factory):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    LaplaceKernel = sumpy["LaplaceKernel"]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = LaplaceKernel(2)

    rng = np.random.default_rng(seed)
    nsources = 70
    ntargets = 3
    sources = obj_array.new_1d([
        actx.from_numpy(
            (-1.0 + 0.03 * rng.standard_normal(nsources)).astype(np.float64)
        ),
        actx.from_numpy(
            (-1.0 + 0.03 * rng.standard_normal(nsources)).astype(np.float64)
        ),
    ])
    targets = obj_array.new_1d([
        actx.from_numpy(
            (1.0 + 0.03 * rng.standard_normal(ntargets)).astype(np.float64)
        ),
        actx.from_numpy(
            (1.0 + 0.03 * rng.standard_normal(ntargets)).astype(np.float64)
        ),
    ])

    classic_result, compat_result = _run_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=4,
        order=3,
        dtype=np.float64,
        weight_seed=seed + 2000,
        m2l_translation_factory=NonFFTM2LTranslationClassFactory(),
    )

    assert np.allclose(compat_result, classic_result, rtol=1e-11, atol=1e-11)


@pytest.mark.opencl
@pytest.mark.parametrize(
    ("dims", "kernel_name", "order", "dtype", "kernel_extra_kwargs", "direct_only"),
    [
        (3, "LaplaceKernel", 3, np.float64, {}, False),
        (2, "HelmholtzKernel", 2, np.complex128, {"k": 0.3}, True),
    ],
)
def test_dual_tree_compatibility_wrangler_matches_sumpy_broader_cases(
    actx_factory, dims, kernel_name, order, dtype, kernel_extra_kwargs, direct_only
):
    sumpy = _import_local_sumpy()
    VolumeTaylorLocalExpansion = sumpy["VolumeTaylorLocalExpansion"]
    VolumeTaylorMultipoleExpansion = sumpy["VolumeTaylorMultipoleExpansion"]
    KernelClass = sumpy[kernel_name]
    NonFFTM2LTranslationClassFactory = sumpy["NonFFTM2LTranslationClassFactory"]
    SumpyPyOpenCLArrayContext = sumpy["PyOpenCLArrayContext"]

    actx = SumpyPyOpenCLArrayContext(actx_factory().queue)
    knl = KernelClass(dims)

    if direct_only:
        nsources = 30
        ntargets = 24
        sources = make_normal_particle_array(actx, nsources, dims, np.float64, seed=71)
        targets = make_normal_particle_array(actx, ntargets, dims, np.float64, seed=72)
        max_particles_in_box = max(nsources, ntargets) + 1
        m2l_factory = None
    else:
        rng = np.random.default_rng(73)
        nsources = 60
        targets = obj_array.new_1d([
            actx.from_numpy(np.array([1.0], dtype=np.float64)) for _ in range(dims)
        ])
        sources = obj_array.new_1d([
            actx.from_numpy(
                (-1.0 + 0.02 * rng.standard_normal(nsources)).astype(np.float64)
            )
            for _ in range(dims)
        ])
        max_particles_in_box = 4
        m2l_factory = NonFFTM2LTranslationClassFactory()

    classic_result, compat_result = _run_sumpy_dual_tree_case(
        actx,
        knl=knl,
        local_expn_class=VolumeTaylorLocalExpansion,
        mpole_expn_class=VolumeTaylorMultipoleExpansion,
        sources=sources,
        targets=targets,
        max_particles_in_box=max_particles_in_box,
        order=order,
        dtype=dtype,
        weight_seed=74,
        kernel_extra_kwargs=kernel_extra_kwargs,
        m2l_translation_factory=m2l_factory,
    )

    assert np.allclose(compat_result, classic_result, rtol=1e-11, atol=1e-11)
