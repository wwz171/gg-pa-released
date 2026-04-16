import numpy as np
import pytest


torch = pytest.importorskip("torch")

from ggpa.systems.phi4 import (
    FixedQGaussianPBCContext,
    GaussianPBCContext,
    _laplacian_spectrum_rfft,
    _precision_spectrum,
    _vp_params_at_tau,
)


class DummyNoiseScheduler:
    def __init__(self):
        self.alpha_bars = torch.linspace(0.95, 0.60, 10)
        self.num_timesteps = len(self.alpha_bars)


def _full_fft_quadratic(log_weights: np.ndarray, s: np.ndarray, L: int) -> float:
    s_hat = np.fft.fftn(s, s=(L, L), norm="ortho")
    return float(-0.5 * np.sum(log_weights * np.abs(s_hat) ** 2))


@pytest.mark.parametrize("L", [5, 6])
def test_gaussian_context_log_prob_matches_full_fft_quadratic_form(L):
    scheduler = DummyNoiseScheduler()
    tau = 0.3
    J = 0.05
    h = 0.1
    s = np.random.default_rng(1234 + L).standard_normal((L, L))

    ctx = GaussianPBCContext(J=J, L=L, noise_scheduler=scheduler, h=h)
    a2, s2 = _vp_params_at_tau(scheduler, tau)
    q_rfft = _precision_spectrum(_laplacian_spectrum_rfft(L), J, a2, s2)

    ky = np.minimum(np.arange(L), L - np.arange(L))
    q_full = q_rfft[:, ky]
    expected = _full_fft_quadratic(q_full, s, L) + float(h / np.sqrt(a2) * np.sum(s))

    assert np.isclose(ctx.log_prob(s, tau), expected, rtol=0.0, atol=1e-10)


def _finite_difference(log_prob_fn, s: np.ndarray, idx: tuple[int, int], eps: float = 1e-6) -> float:
    s_plus = s.copy()
    s_minus = s.copy()
    s_plus[idx] += eps
    s_minus[idx] -= eps
    return (log_prob_fn(s_plus) - log_prob_fn(s_minus)) / (2.0 * eps)


@pytest.mark.parametrize(
    ("context_factory", "tau"),
    [
        (lambda ns, L: GaussianPBCContext(J=0.05, L=L, noise_scheduler=ns, h=0.1), 0.3),
        (
            lambda ns, L: FixedQGaussianPBCContext(
                J=0.05,
                L=L,
                noise_scheduler=ns,
                tau_prod=0.2,
                h=0.1,
            ),
            0.3,
        ),
    ],
)
def test_phi4_context_gradients_match_finite_difference(context_factory, tau):
    scheduler = DummyNoiseScheduler()
    L = 6
    s = np.random.default_rng(2026).standard_normal((L, L))
    ctx = context_factory(scheduler, L)
    grad = np.asarray(ctx.grad_log_prob(s, tau), dtype=np.float64)

    for idx in [(0, 0), (1, 2), (3, 5)]:
        fd = _finite_difference(lambda x: ctx.log_prob(x, tau), s, idx)
        assert np.isclose(grad[idx], fd, rtol=0.0, atol=1e-6)
