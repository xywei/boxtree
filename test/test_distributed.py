import numpy as np
import pyopencl as cl
from mpi4py import MPI
from boxtree.distributed.calculation import DistributedFMMLibExpansionWrangler
from boxtree.distributed import DistributedFMMInfo
import numpy.linalg as la
from boxtree.pyfmmlib_integration import FMMLibExpansionWrangler
import logging
import os
import pytest

# Configure logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logging.getLogger("boxtree.distributed").setLevel(logging.INFO)


def _test_distributed(dims, nsources, ntargets, dtype):

    # Get the current rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize arguments for worker processes
    trav = None
    sources_weights = None
    HELMHOLTZ_K = 0

    # Configure PyOpenCL
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    def fmm_level_to_nterms(tree, level):
        return max(level, 3)

    # Generate particles and run shared-memory parallelism on rank 0
    if rank == 0:

        # Generate random particles and source weights
        from boxtree.tools import make_normal_particle_array as p_normal
        sources = p_normal(queue, nsources, dims, dtype, seed=15)
        targets = p_normal(queue, ntargets, dims, dtype, seed=18)

        from pyopencl.clrandom import PhiloxGenerator
        rng = PhiloxGenerator(queue.context, seed=20)
        sources_weights = rng.uniform(queue, nsources, dtype=np.float64).get()

        from pyopencl.clrandom import PhiloxGenerator
        rng = PhiloxGenerator(queue.context, seed=22)
        target_radii = rng.uniform(
            queue, ntargets, a=0, b=0.05, dtype=np.float64).get()

        # Build the tree and interaction lists
        from boxtree import TreeBuilder
        tb = TreeBuilder(ctx)
        tree, _ = tb(queue, sources, targets=targets, target_radii=target_radii,
                     stick_out_factor=0.25, max_particles_in_box=30, debug=True)

        from boxtree.traversal import FMMTraversalBuilder
        tg = FMMTraversalBuilder(ctx, well_sep_is_n_away=2)
        d_trav, _ = tg(queue, tree, debug=True)
        trav = d_trav.get(queue=queue)

        # Get pyfmmlib expansion wrangler
        wrangler = FMMLibExpansionWrangler(
            trav.tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)

        # Compute FMM using shared memory parallelism
        from boxtree.fmm import drive_fmm
        pot_fmm = drive_fmm(trav, wrangler, sources_weights) * 2 * np.pi

    # Compute FMM using distributed memory parallelism

    def distributed_expansion_wrangler_factory(tree):
        return DistributedFMMLibExpansionWrangler(
            queue, tree, HELMHOLTZ_K, fmm_level_to_nterms=fmm_level_to_nterms)

    distribued_fmm_info = DistributedFMMInfo(
        queue, trav, distributed_expansion_wrangler_factory, comm=comm)
    pot_dfmm = distribued_fmm_info.drive_dfmm(sources_weights)

    if rank == 0:
        error = (la.norm(pot_fmm - pot_dfmm * 2 * np.pi, ord=np.inf) /
                 la.norm(pot_fmm, ord=np.inf))
        print(error)
        assert error < 1e-14


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, dims, nsources, ntargets", [
    (4, 3, 10000, 10000)
])
def test_distributed(num_processes, dims, nsources, ntargets):
    pytest.importorskip("mpi4py")

    newenv = os.environ.copy()
    newenv["PYTEST"] = "1"
    newenv["dims"] = str(dims)
    newenv["nsources"] = str(nsources)
    newenv["ntargets"] = str(ntargets)

    import subprocess
    import sys
    subprocess.run([
        "mpiexec", "-np", str(num_processes),
        "-x", "PYTEST", "-x", "dims", "-x", "nsources", "-x", "ntargets",
        sys.executable, __file__],
        env=newenv,
        check=True
    )


if __name__ == "__main__":

    dtype = np.float64

    if "PYTEST" in os.environ:
        # Run pytest test case
        dims = int(os.environ["dims"])
        nsources = int(os.environ["nsources"])
        ntargets = int(os.environ["ntargets"])

        _test_distributed(dims, nsources, ntargets, dtype)
    else:

        dims = 3
        nsources = 10000
        ntargets = 10000

        _test_distributed(dims, nsources, ntargets, dtype)
