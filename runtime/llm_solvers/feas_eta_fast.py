"""New tutorial algorithms using Codex eta builders; unchanged policy optimizers."""
from functools import partial
from eta_fast import build_eta, build_risk_mass
from feas_two_phase_sparse import feas_iter_two_phase_sparse
from feas_thresh_sparse import feas_iter_two_phase_sparse_thresh


def feas_iter_fast_eta(*args, eta_mode=0, **kwargs):
    return feas_iter_two_phase_sparse(*args, compute_eta=partial(build_eta, mode=eta_mode), **kwargs)


def feas_iter_fast_eta_thresh(*args, eta_mode=0, **kwargs):
    first_pass = True

    def build(*builder_args):
        nonlocal first_pass
        if first_pass:
            first_pass = False
            return build_risk_mass(*builder_args)
        return build_eta(*builder_args, mode=eta_mode)

    return feas_iter_two_phase_sparse_thresh(*args, compute_eta=build, **kwargs)
