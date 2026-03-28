import math

import numpy as np
import pytest

from ggpa.systems.doublewell import (
    CoupledDoubleWell,
    DoubleWellVPForwardProcess,
    finite_t_exact_context_stiffness,
    sample_coupled_equilibrium,
)


def test_sample_coupled_equilibrium_matches_conditional_u_statistics():
    rng = np.random.default_rng(7)
    system = CoupledDoubleWell()

    samples = sample_coupled_equilibrium(4000, system, rng=rng)
    x = samples[:, 0]
    u = samples[:, 1]

    centered_u = u - system.conditional_u_mean(x)

    assert samples.shape == (4000, 2)
    assert abs(float(np.mean(centered_u))) < 0.03
    assert math.isclose(
        float(np.var(centered_u)),
        system.conditional_u_var(),
        rel_tol=0.15,
        abs_tol=0.02,
    )


def test_finite_t_exact_context_stiffness_matches_closed_form():
    torch = pytest.importorskip("torch")

    class DummyScheduler:
        num_timesteps = 2
        alpha_bars = torch.tensor([1.0, 0.81], dtype=torch.float64)

    system = CoupledDoubleWell(k_c=4.0)
    fwd = DoubleWellVPForwardProcess(DummyScheduler())

    t = 1.0
    alpha_t = 0.9
    sigma_t = math.sqrt(0.19)
    expected = system.k_c / (alpha_t**2 - system.k_c * sigma_t**2)

    assert math.isclose(
        finite_t_exact_context_stiffness(t, system=system, forward_process=fwd),
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
