Dual-Tree FMM Prototype
=======================

This page documents the current prototype implementation of a dual-tree FMM
execution path in :mod:`boxtree`.

Overview
--------

The classic :mod:`boxtree` FMM path precomputes traversal lists and then drives
the FMM from those materialized interaction lists. The dual-tree prototype
instead discovers interactions on demand from a simultaneous traversal of the
source and target trees.

The current implementation is intended as a correctness and API prototype. It
is not yet a GPU-native high-performance implementation.

Modules
-------

.. automodule:: boxtree.dual_tree_traversal

.. automodule:: boxtree.dual_tree_fmm

Current Capabilities
--------------------

- End-to-end dual-tree FMM execution with the constant-one dummy wrangler.
- A compatibility wrapper that feeds dual-tree batches into existing list-based
  wranglers through temporary CSR batches.
- A smoke-tested compatibility path for a real :mod:`sumpy` wrangler on a
  direct-only single-box case.
- A smoke-tested compatibility path for a real :mod:`sumpy` wrangler on a
  nontrivial clustered far-field case using non-FFT M2L translations.
- Additional smoke coverage for broader real-:mod:`sumpy` cases, including a
  3D Laplace far-field case and a complex-valued 2D Helmholtz direct case.
- An experimental :mod:`pytential.qbx.fmm.drive_dual_tree_fmm` entry point for
  QBX, with tested constant-one parity for the current experimental path.
- Test coverage for `p2p`, `m2l`, `m2p`, and `p2l` interaction kinds.

Tested Milestones
-----------------

- Dummy constant-one dual-tree execution.
- Compatibility wrapping of a classic list-based wrangler.
- Real-:mod:`sumpy` direct-path compatibility.
- Real-:mod:`sumpy` far-field compatibility.
- Broader real-:mod:`sumpy` compatibility across additional dimensions and a
  complex-valued kernel.
- Experimental QBX dual-tree execution with constant-one parity tests for the
  overall QBX result, focused M2QBXL and L2QBXL stage checks, and matching
  pure dual-tree box locals on QBX-referenced boxes.

Current Limitations
-------------------

- The traversal engine is host-side and not yet device-native.
- The compatibility path allocates temporary CSR structures per interaction
  batch.
- The current `m2l` decomposition is compatible with end-to-end testing and
  the current real-`sumpy` compatibility tests, but is not yet guaranteed to be
  a one-for-one reproduction of the classic precomputed traversal in every
  detail.
- The experimental QBX dual-tree path now matches the constant-one QBX tests
  without the previous QBX-local fallback, but broader QBX kernels and
  nontrivial expansion data still need validation.

Compatibility Invariants
------------------------

The compatibility path in :mod:`boxtree.dual_tree_fmm` currently relies on a
few invariants that were implicit in the classic list-based execution path:

- `m2l` must be presented to downstream wranglers in level-wise CSR form.
- Within a level, target boxes are processed in tree order.
- Within each target box, source boxes are processed in sorted box-number order.

These invariants matter for real-kernel `sumpy`/`pytential` runs because they
affect translation-class grouping and floating-point accumulation order. The
compatibility wrapper therefore buffers `m2l` pairs and replays them one level
at a time instead of issuing arbitrary pair batches directly.

Scheduler As Transition Layer
-----------------------------

The current level-wise `m2l` schedule is an intermediate architecture, not the
intended end state. It exists to preserve classic execution invariants while the
dual-tree path is being made more native.

The intended long-term direction is a fully traversal-free executor that:

- consumes pair discovery directly without materializing even level-wise
  schedules,
- pipelines traversal and translation work more tightly,
- and eventually exploits newer GPU control-flow capabilities, including more
  independent thread progress on NVIDIA hardware.

The current schedule objects should therefore be treated as a compatibility and
migration layer, not as the final execution model.

At runtime, the driver now operates on streaming `m2l` executors. Any remaining
level-wise organization is confined to compatibility executors for wranglers
that still need classic `m2l` invariants. In other words, the schedule is no
longer the runtime execution boundary.

Real-Kernel Regression Workflow
-------------------------------

A small non-FFT real-kernel QBX reproducer lives at
`pytential/examples/dual_tree_qbx_real_kernel_repro.py`.

On `ipa`, run it in the prepared virtualenv with:

.. code-block:: bash

   source /home/xywei/Work/fmm/remote_runs/dual-tree-venv/bin/activate
   python /home/xywei/Work/fmm/pytential/examples/dual_tree_qbx_real_kernel_repro.py

The gated pytest regression in `pytential/test/test_dual_tree_qbx.py` can be
enabled with:

.. code-block:: bash

   PYTENTIAL_RUN_REAL_QBX_DUAL_TREE=1 pytest -q test/test_dual_tree_qbx.py -k real_kernel_nonfft

Intended Direction
------------------

- Replace repeated temporary batch scaffolding with reusable typed containers.
- Tighten semantic parity with the classic traversal where needed.
- Move scheduling and batching toward array-based or GPU-native execution.
