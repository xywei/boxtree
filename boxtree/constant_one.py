"""
.. autoclass:: ConstantOneTreeIndependentDataForWrangler
.. autoclass:: ConstantOneExpansionWrangler
"""

from __future__ import annotations


__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from typing import TYPE_CHECKING

import numpy as np

from boxtree.dual_tree_fmm import (
    CSRBatchAdapter,
    DualTreeExecutionStats,
    DualTreeExpansionWranglerInterface,
    M2LExecutor,
)
from boxtree.fmm import ExpansionWranglerInterface, TreeIndependentDataForWrangler


if TYPE_CHECKING:
    from arraycontext import ArrayContext


# {{{ constant one wrangler


class ConstantOneTreeIndependentDataForWrangler(TreeIndependentDataForWrangler):
    """
    .. automethod:: __init__
    """


class ConstantOneExpansionWrangler(ExpansionWranglerInterface):
    """This implements the 'analytical routines' for a Green's function that is
    constant 1 everywhere. For 'charges' of 'ones', this should get every particle
    a copy of the particle count.
    """

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(pstart, pstart + self.tree.box_target_counts_nonchild[ibox])

    def multipole_expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    def local_expansion_zeros(self, actx=None):
        del actx
        return self.multipole_expansion_zeros()

    def output_zeros(self, actx=None):
        del actx
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def multipole_expansions_view(self, mpole_exps, level):
        # FIXME
        raise NotImplementedError

    def local_expansions_view(self, local_exps, level):
        # FIXME
        raise NotImplementedError

    def form_multipoles(
        self,
        actx: ArrayContext,
        level_start_source_box_nrs,
        source_boxes,
        src_weight_vecs,
    ):
        (src_weights,) = src_weight_vecs
        mpoles = self.multipole_expansion_zeros()

        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    def coarsen_multipoles(
        self,
        actx: ArrayContext,
        level_start_source_parent_box_nrs,
        source_parent_boxes,
        mpoles,
    ):
        tree = self.tree

        # nlevels-1 is the last valid level index
        # nlevels-2 is the last valid level that could have children
        #
        # 3 is the last relevant source_level.
        # 2 is the last relevant target_level.
        # (because no level 1 box will be well-separated from another)
        for source_level in range(tree.nlevels - 1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                target_level : target_level + 2
            ]
            for ibox in source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        mpoles[ibox] += mpoles[child]

        return mpoles

    def eval_direct(
        self,
        actx: ArrayContext,
        target_boxes,
        neighbor_sources_starts,
        neighbor_sources_lists,
        src_weight_vecs,
    ):
        (src_weights,) = src_weight_vecs
        pot = self.output_zeros(None)

        for itgt_box, tgt_ibox in enumerate(target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)

            src_sum = 0
            nsrcs = 0
            start, end = neighbor_sources_starts[itgt_box : itgt_box + 2]
            # print "DIR: %s <- %s" % (tgt_ibox, neighbor_sources_lists[start:end])
            for src_ibox in neighbor_sources_lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                src_sum += np.sum(src_weights[src_pslice])

            pot[tgt_pslice] = src_sum

        return pot

    def multipole_to_local(
        self,
        actx: ArrayContext,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        mpole_exps,
    ):
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box : itgt_box + 2]

            contrib = 0
            # print tgt_ibox, "<-", lists[start:end]
            for src_ibox in lists[start:end]:
                contrib += mpole_exps[src_ibox]

            local_exps[tgt_ibox] += contrib

        return local_exps

    def eval_multipoles(
        self,
        actx: ArrayContext,
        target_boxes_by_source_level,
        from_sep_smaller_nonsiblings_by_level,
        mpole_exps,
    ):
        pot = self.output_zeros(None)

        for level, ssn in enumerate(from_sep_smaller_nonsiblings_by_level):
            for itgt_box, tgt_ibox in enumerate(target_boxes_by_source_level[level]):
                tgt_pslice = self._get_target_slice(tgt_ibox)

                contrib = 0

                start, end = ssn.starts[itgt_box : itgt_box + 2]
                for src_ibox in ssn.lists[start:end]:
                    contrib += mpole_exps[src_ibox]

                pot[tgt_pslice] += contrib

        return pot

    def form_locals(
        self,
        actx: ArrayContext,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        starts,
        lists,
        src_weight_vecs,
    ):
        (src_weights,) = src_weight_vecs
        local_exps = self.local_expansion_zeros()

        for itgt_box, tgt_ibox in enumerate(target_or_target_parent_boxes):
            start, end = starts[itgt_box : itgt_box + 2]

            # print "LIST 4", tgt_ibox, "<-", lists[start:end]
            contrib = 0
            nsrcs = 0
            for src_ibox in lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                nsrcs += src_weights[src_pslice].size

                contrib += np.sum(src_weights[src_pslice])

            local_exps[tgt_ibox] += contrib

        return local_exps

    def refine_locals(
        self,
        actx: ArrayContext,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                target_lev : target_lev + 2
            ]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(
        self, actx: ArrayContext, level_start_target_box_nrs, target_boxes, local_exps
    ):
        pot = self.output_zeros()

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot

    def finalize_potentials(self, actx: ArrayContext, potentials):
        return potentials


class ConstantOneDualTreeExpansionWrangler(DualTreeExpansionWranglerInterface):
    def __init__(self, tree_indep, tree, traversal_engine) -> None:
        super().__init__(tree_indep, tree, traversal_engine)
        self._csr_batch_adapter = CSRBatchAdapter(self.tree.box_id_dtype)
        self.stats = DualTreeExecutionStats()

    def _get_source_slice(self, ibox):
        pstart = self.tree.box_source_starts[ibox]
        return slice(pstart, pstart + self.tree.box_source_counts_nonchild[ibox])

    def _get_target_slice(self, ibox):
        pstart = self.tree.box_target_starts[ibox]
        return slice(pstart, pstart + self.tree.box_target_counts_nonchild[ibox])

    def multipole_expansion_zeros(self):
        return np.zeros(self.tree.nboxes, dtype=np.float64)

    def local_expansion_zeros(self, actx=None):
        del actx
        return self.multipole_expansion_zeros()

    def output_zeros(self, actx=None):
        del actx
        return np.zeros(self.tree.ntargets, dtype=np.float64)

    def reorder_sources(self, source_array):
        return source_array[self.tree.user_source_ids]

    def reorder_potentials(self, potentials):
        return potentials[self.tree.sorted_target_ids]

    def form_multipoles(
        self,
        actx: ArrayContext,
        level_start_source_box_nrs,
        source_boxes,
        src_weight_vecs,
    ):
        del actx
        del level_start_source_box_nrs

        (src_weights,) = src_weight_vecs
        mpoles = self.multipole_expansion_zeros()

        for ibox in source_boxes:
            pslice = self._get_source_slice(ibox)
            mpoles[ibox] += np.sum(src_weights[pslice])

        return mpoles

    def coarsen_multipoles(
        self,
        actx: ArrayContext,
        level_start_source_parent_box_nrs,
        source_parent_boxes,
        mpoles,
    ):
        del actx

        tree = self.tree
        for source_level in range(tree.nlevels - 1, 2, -1):
            target_level = source_level - 1
            start, stop = level_start_source_parent_box_nrs[
                target_level : target_level + 2
            ]
            for ibox in source_parent_boxes[start:stop]:
                for child in tree.box_child_ids[:, ibox]:
                    if child:
                        mpoles[ibox] += mpoles[child]

        return mpoles

    def eval_direct_batch(
        self, actx: ArrayContext, pair_batch, src_weight_vecs, potentials
    ):
        del actx

        (src_weights,) = src_weight_vecs

        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                potentials[tgt_pslice] += np.sum(src_weights[src_pslice])

        return potentials

    def multipole_to_local_batch(
        self, actx: ArrayContext, pair_batch, mpole_exps, local_exps
    ):
        del actx

        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                local_exps[tgt_ibox] += mpole_exps[src_ibox]

        return local_exps

    def execute_levelwise_m2l_schedule(self, actx, schedule, mpole_exps, local_exps):
        del actx

        for level_batch in schedule.level_batches:
            csr_batch = level_batch.csr_batch
            for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
                start, end = csr_batch.starts[itgt_box : itgt_box + 2]
                for src_ibox in csr_batch.lists[start:end]:
                    local_exps[tgt_ibox] += mpole_exps[src_ibox]

        return local_exps

    def execute_levelwise_m2l_batch(
        self, actx, level, csr_batch, mpole_exps, local_exps
    ):
        del actx, level

        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                local_exps[tgt_ibox] += mpole_exps[src_ibox]

        return local_exps

    def uses_native_levelwise_m2l(self):
        return True

    def uses_traversal_free_m2l(self):
        return True

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

    def make_m2l_executor(self, actx, mpole_exps, local_exps):
        del actx
        return _TraversalFreeConstantOneM2LExecutor(self, mpole_exps, local_exps)

    def eval_multipoles(
        self,
        actx: ArrayContext,
        target_boxes_by_source_level,
        from_sep_smaller_nonsiblings_by_level,
        mpole_exps,
    ):
        raise NotImplementedError

    def eval_multipoles_batch(
        self, actx: ArrayContext, pair_batch, mpole_exps, potentials
    ):
        del actx

        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                potentials[tgt_pslice] += mpole_exps[src_ibox]

        return potentials

    def form_locals_batch(
        self, actx: ArrayContext, pair_batch, src_weight_vecs, local_exps
    ):
        del actx

        (src_weights,) = src_weight_vecs
        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                local_exps[tgt_ibox] += np.sum(src_weights[src_pslice])

        return local_exps

    def refine_locals(
        self,
        actx: ArrayContext,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        del actx

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                target_lev : target_lev + 2
            ]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(
        self, actx: ArrayContext, level_start_target_box_nrs, target_boxes, local_exps
    ):
        del actx
        del level_start_target_box_nrs

        pot = self.output_zeros(None)

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot

    def finalize_potentials(self, actx: ArrayContext, potentials):
        del actx
        return potentials


class _TraversalFreeConstantOneM2LExecutor(M2LExecutor):
    def __init__(self, wrangler, mpole_exps, local_exps):
        self.wrangler = wrangler
        self.mpole_exps = mpole_exps
        self.local_exps = local_exps

    def consume_pair_batch(self, pair_batch):
        self.wrangler.record_m2l_pair_count(len(pair_batch.target_boxes))
        self.local_exps = self.wrangler.multipole_to_local_batch(
            None, pair_batch, self.mpole_exps, self.local_exps
        )

    def finalize(self, local_exps):
        del local_exps
        self.wrangler.record_used_levelwise_m2l_schedule(False)
        return self.local_exps

    def eval_multipoles(
        self,
        actx: ArrayContext,
        target_boxes_by_source_level,
        from_sep_smaller_nonsiblings_by_level,
        mpole_exps,
    ):
        raise NotImplementedError

    def eval_multipoles_batch(
        self, actx: ArrayContext, pair_batch, mpole_exps, potentials
    ):
        del actx

        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            tgt_pslice = self._get_target_slice(tgt_ibox)
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                potentials[tgt_pslice] += mpole_exps[src_ibox]

        return potentials

    def form_locals_batch(
        self, actx: ArrayContext, pair_batch, src_weight_vecs, local_exps
    ):
        del actx

        (src_weights,) = src_weight_vecs
        csr_batch = self._csr_batch_adapter.pair_batch_to_csr(pair_batch)
        for itgt_box, tgt_ibox in enumerate(csr_batch.target_boxes):
            start, end = csr_batch.starts[itgt_box : itgt_box + 2]
            for src_ibox in csr_batch.lists[start:end]:
                src_pslice = self._get_source_slice(src_ibox)
                local_exps[tgt_ibox] += np.sum(src_weights[src_pslice])

        return local_exps

    def refine_locals(
        self,
        actx: ArrayContext,
        level_start_target_or_target_parent_box_nrs,
        target_or_target_parent_boxes,
        local_exps,
    ):
        del actx

        for target_lev in range(1, self.tree.nlevels):
            start, stop = level_start_target_or_target_parent_box_nrs[
                target_lev : target_lev + 2
            ]
            for ibox in target_or_target_parent_boxes[start:stop]:
                local_exps[ibox] += local_exps[self.tree.box_parent_ids[ibox]]

        return local_exps

    def eval_locals(
        self, actx: ArrayContext, level_start_target_box_nrs, target_boxes, local_exps
    ):
        del actx
        del level_start_target_box_nrs

        pot = self.output_zeros(None)

        for ibox in target_boxes:
            tgt_pslice = self._get_target_slice(ibox)
            pot[tgt_pslice] += local_exps[ibox]

        return pot

    def finalize_potentials(self, actx: ArrayContext, potentials):
        del actx
        return potentials


# }}}

# vim: foldmethod=marker:filetype=pyopencl
