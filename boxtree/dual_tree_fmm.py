"""Prototype dual-tree FMM driver and compatibility adapters.

This module provides a sibling execution path to :mod:`boxtree.fmm` that uses
on-the-fly dual-tree traversal batches instead of a fully materialized
`FMMTraversalInfo` object.
"""

from __future__ import annotations


from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from boxtree.fmm import TreeIndependentDataForWrangler
from boxtree.dual_tree_traversal import PairBatch


@dataclass(frozen=True)
class DualTreeLevelData:
    source_boxes: np.ndarray
    level_start_source_box_nrs: np.ndarray
    source_parent_boxes: np.ndarray
    level_start_source_parent_box_nrs: np.ndarray
    target_boxes: np.ndarray
    level_start_target_box_nrs: np.ndarray
    target_or_target_parent_boxes: np.ndarray
    level_start_target_or_target_parent_box_nrs: np.ndarray


@dataclass(frozen=True)
class CSRInteractionBatch:
    target_boxes: np.ndarray
    starts: np.ndarray
    lists: np.ndarray


@dataclass(frozen=True)
class CSRLevelListBatch:
    starts: np.ndarray
    lists: np.ndarray


@dataclass(frozen=True)
class M2LLevelSchedule:
    level: int
    csr_batch: CSRInteractionBatch


@dataclass(frozen=True)
class LevelwiseM2LSchedule:
    level_batches: tuple[M2LLevelSchedule, ...]


class M2LExecutor(ABC):
    @abstractmethod
    def consume_pair_batch(self, pair_batch):
        pass

    @abstractmethod
    def finalize(self, local_exps):
        pass

    def consume_level_end(self, level, local_exps):
        del level
        return local_exps


def execute_levelwise_m2l_schedule(
    actx,
    tree,
    classic_wrangler,
    to_device_array,
    schedule: LevelwiseM2LSchedule,
    mpole_exps,
    local_exps,
):
    for level_batch in schedule.level_batches:
        level_starts = _build_level_starts(tree, level_batch.csr_batch.target_boxes)
        local_exps = local_exps + classic_wrangler.multipole_to_local(
            actx,
            to_device_array(actx, level_starts),
            to_device_array(actx, level_batch.csr_batch.target_boxes),
            to_device_array(actx, level_batch.csr_batch.starts),
            to_device_array(actx, level_batch.csr_batch.lists),
            mpole_exps,
        )

    return local_exps


@dataclass
class DualTreeExecutionStats:
    traversal_seconds: float = 0.0
    m2l_finalize_seconds: float = 0.0
    m2l_pair_count: int = 0
    m2l_level_count: int = 0
    used_levelwise_m2l_schedule: bool = False


class LevelwiseCSRAccumulator:
    def __init__(self, box_id_dtype) -> None:
        self.box_id_dtype = np.dtype(box_id_dtype)
        self.level_to_targets: dict[int, dict[int, list[int]]] = {}

    def add_pairs(self, level: int, source_boxes, target_boxes) -> None:
        level_targets = self.level_to_targets.setdefault(level, {})
        for src, tgt in zip(source_boxes, target_boxes, strict=True):
            level_targets.setdefault(int(tgt), []).append(int(src))

    def iter_level_batches(self):
        for level in sorted(self.level_to_targets):
            level_targets = self.level_to_targets[level]
            target_box_order = sorted(level_targets)
            starts = np.empty(len(target_box_order) + 1, dtype=self.box_id_dtype)
            starts[0] = 0

            flat_sources: list[int] = []
            for i, target_box in enumerate(target_box_order, start=1):
                flat_sources.extend(sorted(level_targets[target_box]))
                starts[i] = len(flat_sources)

            yield (
                level,
                CSRInteractionBatch(
                    target_boxes=np.array(target_box_order, dtype=self.box_id_dtype),
                    starts=starts,
                    lists=np.array(flat_sources, dtype=self.box_id_dtype),
                ),
            )

    def as_schedule(self) -> LevelwiseM2LSchedule:
        return LevelwiseM2LSchedule(
            level_batches=tuple(
                M2LLevelSchedule(level=level, csr_batch=csr_batch)
                for level, csr_batch in self.iter_level_batches()
            )
        )

    def clear(self) -> None:
        self.level_to_targets.clear()


class CurrentLevelCSRAccumulator:
    def __init__(self, box_id_dtype) -> None:
        self.box_id_dtype = np.dtype(box_id_dtype)
        self.level: int | None = None
        self.targets: dict[int, list[int]] = {}

    def add_pair(self, level: int, source_box: int, target_box: int) -> None:
        if self.level is None:
            self.level = level
        elif self.level != level:
            raise ValueError("current-level accumulator received a different level")
        self.targets.setdefault(target_box, []).append(source_box)

    def clear(self) -> None:
        self.level = None
        self.targets.clear()

    def is_empty(self) -> bool:
        return self.level is None or not self.targets

    def to_level_batch(self) -> M2LLevelSchedule:
        if self.level is None:
            raise ValueError("current-level accumulator is empty")

        target_box_order = sorted(self.targets)
        starts = np.empty(len(target_box_order) + 1, dtype=self.box_id_dtype)
        starts[0] = 0
        flat_sources: list[int] = []
        for i, target_box in enumerate(target_box_order, start=1):
            flat_sources.extend(sorted(self.targets[target_box]))
            starts[i] = len(flat_sources)

        return M2LLevelSchedule(
            level=self.level,
            csr_batch=CSRInteractionBatch(
                target_boxes=np.array(target_box_order, dtype=self.box_id_dtype),
                starts=starts,
                lists=np.array(flat_sources, dtype=self.box_id_dtype),
            ),
        )


def build_levelwise_m2l_schedule_from_traversal(
    tree,
    traversal_engine,
    *,
    pair_consumer=None,
    m2l_pair_counter=None,
):
    m2l_accumulator = LevelwiseCSRAccumulator(tree.box_id_dtype)
    traversal_start_time = perf_counter()

    def visit_pair(interaction_kind, source_box, target_box, target_level):
        if interaction_kind == "m2l":
            m2l_accumulator.add_pairs(
                target_level,
                np.array([source_box], dtype=tree.box_id_dtype),
                np.array([target_box], dtype=tree.box_id_dtype),
            )
            if m2l_pair_counter is not None:
                m2l_pair_counter(1)
        elif pair_consumer is not None:
            pair_consumer(
                PairBatch(
                    interaction_kind,
                    np.array([source_box], dtype=tree.box_id_dtype),
                    np.array([target_box], dtype=tree.box_id_dtype),
                )
            )

    traversal_engine.walk_streaming(tree, pair_visitor=visit_pair)

    return m2l_accumulator.as_schedule(), perf_counter() - traversal_start_time


class DualTreeExpansionWranglerInterface(ABC):
    def __init__(
        self, tree_indep: TreeIndependentDataForWrangler, tree, traversal_engine
    ) -> None:
        self.tree_indep = tree_indep
        self._tree = tree
        self.traversal_engine = traversal_engine

    @property
    def tree(self):
        return self._tree

    @abstractmethod
    def output_zeros(self, actx):
        pass

    @abstractmethod
    def local_expansion_zeros(self, actx):
        pass

    @abstractmethod
    def reorder_sources(self, source_array):
        pass

    @abstractmethod
    def reorder_potentials(self, potentials):
        pass

    @abstractmethod
    def form_multipoles(
        self, actx, level_start_source_box_nrs, source_boxes, src_weight_vecs
    ):
        pass

    @abstractmethod
    def coarsen_multipoles(
        self, actx, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        pass

    @abstractmethod
    def eval_direct_batch(self, actx, pair_batch, src_weight_vecs, potentials):
        pass

    @abstractmethod
    def multipole_to_local_batch(self, actx, pair_batch, mpole_exps, local_exps):
        pass

    @abstractmethod
    def eval_multipoles_batch(self, actx, pair_batch, mpole_exps, potentials):
        pass

    @abstractmethod
    def form_locals_batch(self, actx, pair_batch, src_weight_vecs, local_exps):
        pass

    @abstractmethod
    def refine_locals(
        self,
        actx,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        pass

    @abstractmethod
    def eval_locals(self, actx, level_start_target_box_nrs, target_boxes, local_exps):
        pass

    @abstractmethod
    def finalize_potentials(self, actx, potentials):
        pass

    def make_m2l_executor(self, actx, mpole_exps, local_exps):
        del actx, mpole_exps, local_exps
        raise NotImplementedError

    def execute_levelwise_m2l_batch(
        self, actx, level, csr_batch, mpole_exps, local_exps
    ):
        return self.execute_levelwise_m2l_schedule(
            actx,
            LevelwiseM2LSchedule((M2LLevelSchedule(level=level, csr_batch=csr_batch),)),
            mpole_exps,
            local_exps,
        )

    def execute_levelwise_m2l_schedule(self, actx, schedule, mpole_exps, local_exps):
        del schedule
        return self.finalize_multipole_to_local(actx, mpole_exps, local_exps)

    def uses_native_levelwise_m2l(self):
        return False

    def uses_traversal_free_m2l(self):
        return False

    def finalize_multipole_to_local(self, actx, mpole_exps, local_exps):
        del actx, mpole_exps
        return local_exps

    def get_execution_stats(self):
        raise NotImplementedError

    def reset_execution_stats(self):
        raise NotImplementedError

    def record_traversal_seconds(self, traversal_seconds):
        del traversal_seconds
        return None

    def record_m2l_pair_count(self, pair_count):
        del pair_count
        return None

    def record_m2l_level_count(self, level_count):
        del level_count
        return None

    def record_used_levelwise_m2l_schedule(self, used_schedule):
        del used_schedule
        return None


class CSRBatchAdapter:
    def __init__(self, box_id_dtype) -> None:
        self.box_id_dtype = np.dtype(box_id_dtype)

    def pair_batch_to_csr(self, pair_batch):
        source_boxes = np.asarray(pair_batch.source_boxes, dtype=self.box_id_dtype)
        target_boxes = np.asarray(pair_batch.target_boxes, dtype=self.box_id_dtype)

        grouped_sources: dict[int, list[int]] = {}

        for source_box, target_box in zip(source_boxes, target_boxes, strict=True):
            target_box_int = int(target_box)
            if target_box_int not in grouped_sources:
                grouped_sources[target_box_int] = []
            grouped_sources[target_box_int].append(int(source_box))

        target_box_order = sorted(grouped_sources)

        target_box_array = np.array(target_box_order, dtype=self.box_id_dtype)
        starts = np.empty(len(target_box_order) + 1, dtype=self.box_id_dtype)
        starts[0] = 0

        flat_sources: list[int] = []
        for i, target_box in enumerate(target_box_order, start=1):
            flat_sources.extend(sorted(grouped_sources[target_box]))
            starts[i] = len(flat_sources)

        lists = np.array(flat_sources, dtype=self.box_id_dtype)
        return CSRInteractionBatch(target_box_array, starts, lists)


class DualTreeCompatibilityWrangler(DualTreeExpansionWranglerInterface):
    def __init__(
        self,
        tree_indep: TreeIndependentDataForWrangler,
        tree,
        traversal_engine,
        classic_wrangler,
    ) -> None:
        super().__init__(tree_indep, tree, traversal_engine)
        self.classic_wrangler = classic_wrangler
        self._csr_batch_adapter = CSRBatchAdapter(self.tree.box_id_dtype)
        self.stats = DualTreeExecutionStats()
        self._classic_uses_device_arrays = not isinstance(
            self.classic_wrangler.tree.box_source_starts, np.ndarray
        )
        self._deferred_m2l = LevelwiseCSRAccumulator(self.tree.box_id_dtype)

    def output_zeros(self, actx):
        return self.classic_wrangler.output_zeros(actx)

    def local_expansion_zeros(self, actx):
        return self.classic_wrangler.local_expansion_zeros(actx)

    def reorder_sources(self, source_array):
        return self.classic_wrangler.reorder_sources(source_array)

    def reorder_potentials(self, potentials):
        return self.classic_wrangler.reorder_potentials(potentials)

    def _to_device_array(self, actx, array):
        if self._classic_uses_device_arrays and isinstance(array, np.ndarray):
            return actx.from_numpy(array)
        return array

    def _to_device_level_list_batch(self, actx, batch):
        if not self._classic_uses_device_arrays:
            return batch
        return CSRLevelListBatch(
            starts=self._to_device_array(actx, batch.starts),
            lists=self._to_device_array(actx, batch.lists),
        )

    def form_multipoles(
        self, actx, level_start_source_box_nrs, source_boxes, src_weight_vecs
    ):
        return self.classic_wrangler.form_multipoles(
            actx,
            self._to_device_array(actx, level_start_source_box_nrs),
            self._to_device_array(actx, source_boxes),
            src_weight_vecs,
        )

    def coarsen_multipoles(
        self, actx, level_start_source_parent_box_nrs, source_parent_boxes, mpoles
    ):
        return self.classic_wrangler.coarsen_multipoles(
            actx,
            self._to_device_array(actx, level_start_source_parent_box_nrs),
            self._to_device_array(actx, source_parent_boxes),
            mpoles,
        )

    def eval_direct_batch(self, actx, pair_batch, src_weight_vecs, potentials):
        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        return potentials + self.classic_wrangler.eval_direct(
            actx,
            self._to_device_array(actx, csr_batch.target_boxes),
            self._to_device_array(actx, csr_batch.starts),
            self._to_device_array(actx, csr_batch.lists),
            src_weight_vecs,
        )

    def multipole_to_local_batch(self, actx, pair_batch, mpole_exps, local_exps):
        del actx, mpole_exps
        if len(pair_batch.target_boxes) == 0:
            return local_exps

        # For real-kernel M2L, downstream wranglers assume a level-wise CSR layout
        # that is processed in one call per level. Preserving that implicit
        # invariant avoids changing translation-class grouping and accumulation
        # order across the boxtree/sumpy/pytential boundary.
        level = int(self.tree.box_levels[int(pair_batch.target_boxes[0])])
        self._deferred_m2l.add_pairs(
            level, pair_batch.source_boxes, pair_batch.target_boxes
        )
        self.stats.m2l_pair_count += len(pair_batch.target_boxes)
        return local_exps

    def finalize_multipole_to_local(self, actx, mpole_exps, local_exps):
        start_time = perf_counter()
        schedule = self._deferred_m2l.as_schedule()
        local_exps = self.execute_levelwise_m2l_schedule(
            actx,
            schedule,
            mpole_exps,
            local_exps,
        )
        self._deferred_m2l.clear()
        self.stats.m2l_level_count += len(schedule.level_batches)
        self.stats.m2l_finalize_seconds += perf_counter() - start_time
        return local_exps

    def execute_levelwise_m2l_schedule(self, actx, schedule, mpole_exps, local_exps):
        return execute_levelwise_m2l_schedule(
            actx,
            self.tree,
            self.classic_wrangler,
            self._to_device_array,
            schedule,
            mpole_exps,
            local_exps,
        )

    def execute_levelwise_m2l_batch(
        self, actx, level, csr_batch, mpole_exps, local_exps
    ):
        del level
        level_starts = _build_level_starts(self.tree, csr_batch.target_boxes)
        return local_exps + self.classic_wrangler.multipole_to_local(
            actx,
            self._to_device_array(actx, level_starts),
            self._to_device_array(actx, csr_batch.target_boxes),
            self._to_device_array(actx, csr_batch.starts),
            self._to_device_array(actx, csr_batch.lists),
            mpole_exps,
        )

    def uses_native_levelwise_m2l(self):
        return False

    def eval_multipoles_batch(self, actx, pair_batch, mpole_exps, potentials):
        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        source_level = int(self.tree.box_levels[pair_batch.source_boxes[0]])

        list3 = self._to_device_level_list_batch(
            actx, CSRLevelListBatch(starts=csr_batch.starts, lists=csr_batch.lists)
        )
        empty_list3 = self._to_device_level_list_batch(
            actx,
            CSRLevelListBatch(
                starts=np.array([0], dtype=self.tree.box_id_dtype),
                lists=np.empty(0, dtype=self.tree.box_id_dtype),
            ),
        )

        result = self.classic_wrangler.eval_multipoles(
            actx,
            [self._to_device_array(actx, np.empty(0, dtype=self.tree.box_id_dtype))]
            * source_level
            + [self._to_device_array(actx, csr_batch.target_boxes)]
            + [self._to_device_array(actx, np.empty(0, dtype=self.tree.box_id_dtype))]
            * (self.tree.nlevels - source_level - 1),
            [empty_list3] * source_level
            + [list3]
            + [empty_list3] * (self.tree.nlevels - source_level - 1),
            mpole_exps,
        )
        return potentials + result

    def form_locals_batch(self, actx, pair_batch, src_weight_vecs, local_exps):
        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        level_starts = _build_level_starts(self.tree, csr_batch.target_boxes)
        return local_exps + self.classic_wrangler.form_locals(
            actx,
            self._to_device_array(actx, level_starts),
            self._to_device_array(actx, csr_batch.target_boxes),
            self._to_device_array(actx, csr_batch.starts),
            self._to_device_array(actx, csr_batch.lists),
            src_weight_vecs,
        )

    def refine_locals(
        self,
        actx,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        return self.classic_wrangler.refine_locals(
            actx,
            self._to_device_array(actx, level_start_target_or_target_parent_box_nrs),
            self._to_device_array(actx, target_or_target_parent_boxes),
            local_exps,
        )

    def eval_locals(self, actx, level_start_target_box_nrs, target_boxes, local_exps):
        return self.classic_wrangler.eval_locals(
            actx,
            self._to_device_array(actx, level_start_target_box_nrs),
            self._to_device_array(actx, target_boxes),
            local_exps,
        )

    def finalize_potentials(self, actx, potentials):
        return self.classic_wrangler.finalize_potentials(actx, potentials)

    def get_execution_stats(self):
        return DualTreeExecutionStats(**self.stats.__dict__)

    def reset_execution_stats(self):
        self.stats = DualTreeExecutionStats()

    def record_traversal_seconds(self, traversal_seconds):
        self.stats.traversal_seconds += traversal_seconds

    def record_m2l_pair_count(self, pair_count):
        self.stats.m2l_pair_count += pair_count

    def record_m2l_level_count(self, level_count):
        self.stats.m2l_level_count += level_count

    def record_used_levelwise_m2l_schedule(self, used_schedule):
        self.stats.used_levelwise_m2l_schedule = used_schedule

    def build_levelwise_m2l_schedule(self):
        return self._deferred_m2l.as_schedule()

    def make_m2l_executor(self, actx, mpole_exps, local_exps):
        return _BufferedCompatibilityM2LExecutor(self, actx, mpole_exps, local_exps)


class _BufferedCompatibilityM2LExecutor(M2LExecutor):
    def __init__(self, wrangler, actx, mpole_exps, local_exps):
        self.wrangler = wrangler
        self.actx = actx
        self.mpole_exps = mpole_exps
        self.local_exps = local_exps
        self.accumulator = CurrentLevelCSRAccumulator(wrangler.tree.box_id_dtype)

    def consume_pair_batch(self, pair_batch):
        if len(pair_batch.target_boxes) == 0:
            return
        level = int(self.wrangler.tree.box_levels[int(pair_batch.target_boxes[0])])
        for src, tgt in zip(
            pair_batch.source_boxes, pair_batch.target_boxes, strict=True
        ):
            self.accumulator.add_pair(level, int(src), int(tgt))
            self.wrangler.record_m2l_pair_count(1)

    def _flush_level(self, level):
        if self.accumulator.is_empty():
            return
        if self.accumulator.level != level:
            return

        level_batch = self.accumulator.to_level_batch()

        result = self.wrangler.execute_levelwise_m2l_batch(
            self.actx,
            level,
            level_batch.csr_batch,
            self.mpole_exps,
            self.local_exps,
        )
        self.local_exps = result
        self.wrangler.record_used_levelwise_m2l_schedule(True)
        self.wrangler.record_m2l_level_count(1)
        self.accumulator.clear()

    def consume_level_end(self, level, local_exps):
        del local_exps
        self._flush_level(level)
        return self.local_exps

    def finalize(self, local_exps):
        del local_exps
        if not self.accumulator.is_empty():
            self._flush_level(self.accumulator.level)
        return self.wrangler.finalize_multipole_to_local(
            self.actx, self.mpole_exps, self.local_exps
        )


def drive_dual_tree_fmm(
    actx,
    wrangler: DualTreeExpansionWranglerInterface,
    src_weight_vecs,
    *,
    global_src_idx_all_ranks=None,
    global_tgt_idx_all_ranks=None,
):
    del global_src_idx_all_ranks
    del global_tgt_idx_all_ranks

    level_data = _build_level_data(wrangler.tree)

    src_weight_vecs = [wrangler.reorder_sources(weight) for weight in src_weight_vecs]

    mpole_exps = wrangler.form_multipoles(
        actx,
        level_data.level_start_source_box_nrs,
        level_data.source_boxes,
        src_weight_vecs,
    )

    mpole_exps = wrangler.coarsen_multipoles(
        actx,
        level_data.level_start_source_parent_box_nrs,
        level_data.source_parent_boxes,
        mpole_exps,
    )

    local_exps = wrangler.local_expansion_zeros(actx)
    potentials = wrangler.output_zeros(actx)
    m2l_executor = wrangler.make_m2l_executor(actx, mpole_exps, local_exps)

    if wrangler.uses_traversal_free_m2l():
        traversal_start_time = perf_counter()
        current_m2l_level = None

        def visit_pair(interaction_kind, source_box, target_box, target_level):
            nonlocal current_m2l_level, local_exps, potentials

            if current_m2l_level is not None and target_level != current_m2l_level:
                local_exps = m2l_executor.consume_level_end(
                    current_m2l_level, local_exps
                )
            current_m2l_level = target_level

            pair_batch = PairBatch(
                interaction_kind,
                np.array([source_box], dtype=wrangler.tree.box_id_dtype),
                np.array([target_box], dtype=wrangler.tree.box_id_dtype),
            )

            if interaction_kind == "p2p":
                potentials = wrangler.eval_direct_batch(
                    actx, pair_batch, src_weight_vecs, potentials
                )
            elif interaction_kind == "m2l":
                m2l_executor.consume_pair_batch(pair_batch)
            elif interaction_kind == "m2p":
                potentials = wrangler.eval_multipoles_batch(
                    actx, pair_batch, mpole_exps, potentials
                )
            elif interaction_kind == "p2l":
                local_exps = wrangler.form_locals_batch(
                    actx, pair_batch, src_weight_vecs, local_exps
                )
            else:
                raise ValueError(f"unknown interaction kind: {interaction_kind}")

        wrangler.traversal_engine.walk_streaming(wrangler.tree, pair_visitor=visit_pair)

        if current_m2l_level is not None:
            local_exps = m2l_executor.consume_level_end(current_m2l_level, local_exps)

        wrangler.record_traversal_seconds(perf_counter() - traversal_start_time)
        wrangler.record_used_levelwise_m2l_schedule(False)
        local_exps = m2l_executor.finalize(local_exps)
    else:
        traversal_start_time = perf_counter()
        current_m2l_level = None

        def visit_pair(interaction_kind, source_box, target_box, target_level):
            nonlocal current_m2l_level, local_exps, potentials

            if current_m2l_level is not None and target_level != current_m2l_level:
                local_exps = m2l_executor.consume_level_end(
                    current_m2l_level, local_exps
                )
            current_m2l_level = target_level

            pair_batch = PairBatch(
                interaction_kind,
                np.array([source_box], dtype=wrangler.tree.box_id_dtype),
                np.array([target_box], dtype=wrangler.tree.box_id_dtype),
            )

            if interaction_kind == "p2p":
                potentials = wrangler.eval_direct_batch(
                    actx, pair_batch, src_weight_vecs, potentials
                )
            elif interaction_kind == "m2l":
                m2l_executor.consume_pair_batch(pair_batch)
            elif interaction_kind == "m2p":
                potentials = wrangler.eval_multipoles_batch(
                    actx, pair_batch, mpole_exps, potentials
                )
            elif interaction_kind == "p2l":
                local_exps = wrangler.form_locals_batch(
                    actx, pair_batch, src_weight_vecs, local_exps
                )
            else:
                raise ValueError(f"unknown interaction kind: {interaction_kind}")

        wrangler.traversal_engine.walk_streaming(wrangler.tree, pair_visitor=visit_pair)

        if current_m2l_level is not None:
            local_exps = m2l_executor.consume_level_end(current_m2l_level, local_exps)

        wrangler.record_traversal_seconds(perf_counter() - traversal_start_time)
        local_exps = m2l_executor.finalize(local_exps)

    local_exps = wrangler.refine_locals(
        actx,
        level_data.level_start_target_or_target_parent_box_nrs,
        level_data.target_or_target_parent_boxes,
        local_exps,
    )

    potentials = potentials + wrangler.eval_locals(
        actx,
        level_data.level_start_target_box_nrs,
        level_data.target_boxes,
        local_exps,
    )

    return wrangler.finalize_potentials(
        actx,
        wrangler.reorder_potentials(potentials),
    )


def _build_level_data(tree) -> DualTreeLevelData:
    source_boxes = np.flatnonzero(tree.box_flags & 1).astype(tree.box_id_dtype)
    source_parent_boxes = np.flatnonzero(tree.box_flags & (1 << 2)).astype(
        tree.box_id_dtype
    )
    target_boxes = np.flatnonzero(tree.box_flags & (1 << 1)).astype(tree.box_id_dtype)
    target_or_target_parent_boxes = np.flatnonzero(
        tree.box_flags & ((1 << 1) | (1 << 3))
    ).astype(tree.box_id_dtype)

    return DualTreeLevelData(
        source_boxes=source_boxes,
        level_start_source_box_nrs=_build_level_starts(tree, source_boxes),
        source_parent_boxes=source_parent_boxes,
        level_start_source_parent_box_nrs=_build_level_starts(
            tree, source_parent_boxes
        ),
        target_boxes=target_boxes,
        level_start_target_box_nrs=_build_level_starts(tree, target_boxes),
        target_or_target_parent_boxes=target_or_target_parent_boxes,
        level_start_target_or_target_parent_box_nrs=_build_level_starts(
            tree, target_or_target_parent_boxes
        ),
    )


def _build_level_starts(tree, boxes: np.ndarray) -> np.ndarray:
    result = np.empty(tree.nlevels + 1, dtype=tree.box_id_dtype)
    next_index = 0
    for level in range(tree.nlevels):
        result[level] = next_index
        while (
            next_index < len(boxes) and int(tree.box_levels[boxes[next_index]]) == level
        ):
            next_index += 1
    result[tree.nlevels] = len(boxes)
    return result
