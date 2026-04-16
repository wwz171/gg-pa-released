"""High-level drivers that wrap FixedDiffusionTimeKernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np

from ggpa.core.kernel import FixedDiffusionTimeKernel
from ggpa.core.state import State
from ggpa.core.errors import ConfigurationError, NotSupportedError
from ggpa.utils.utils import rng_from_seed


@dataclass
class FixedDiffusionTimeSampler:
    """Run a fixed-diffusion-time kernel for a number of steps.

    Attributes:
        kernel: The FixedDiffusionTimeKernel to use.
        t_diff: The fixed diffusion time.
    """

    kernel: FixedDiffusionTimeKernel
    t_diff: float

    def run(self, state: State, n_steps: int) -> Tuple[State, List]:
        """Run the sampler for n_steps iterations.

        Args:
            state: Initial state.
            n_steps: Number of Gibbs iterations to perform.

        Returns:
            Tuple of (final_state, list_of_diagnostics).
        """
        diagnostics = []
        current = state
        for _ in range(n_steps):
            current, diag = self.kernel.step(current, self.t_diff)
            diagnostics.append(diag)
        return current, diagnostics


@dataclass
class AnnealingDriver:
    """Run a kernel with a provided t_diff schedule.

    Attributes:
        kernel: The FixedDiffusionTimeKernel to use.
    """

    kernel: FixedDiffusionTimeKernel

    def run(self, state: State, schedule: Iterable[float]) -> Tuple[State, List]:
        """Run the kernel with an annealing schedule.

        Args:
            state: Initial state.
            schedule: Iterable of t_diff values (e.g., from linear_schedule).

        Returns:
            Tuple of (final_state, list_of_diagnostics).
        """
        diagnostics = []
        current = state
        for t_diff in schedule:
            current, diag = self.kernel.step(current, float(t_diff))
            diagnostics.append(diag)
        return current, diagnostics


@dataclass
class ReplicaExchangeDriver:
    """Experimental generic replica-exchange driver.

    This helper is intentionally disabled in the public release.
    The current research code uses system-specific replica-exchange
    implementations that explicitly manage the per-replica latent
    ``x`` states needed for correct swap bookkeeping:

    - ``ggpa.systems.phi4.LatticeRERunner``
    - ``ggpa.systems.alanine_dipeptide.AlanineReplicaExchange``

    Attributes:
        kernel: The FixedDiffusionTimeKernel shared by all replicas.
        t_diffs: List of t_diff values, one per replica (ascending order recommended).
        swap_interval: Attempt swaps every this many blocks.
        master_seed: Optional seed for reproducible swap decisions.
        rng: NumPy random generator (created from master_seed if None).
    """

    kernel: FixedDiffusionTimeKernel
    t_diffs: List[float]
    swap_interval: int = 5
    master_seed: Optional[int] = None
    rng: Optional[np.random.Generator] = None

    def run(self, states: List[State], n_blocks: int, inner_steps: int) -> Tuple[List[State], List[dict]]:
        """Run replica exchange sampling.

        Args:
            states: List of initial states, one per replica.
            n_blocks: Number of outer blocks.
            inner_steps: Number of kernel steps per replica per block.

        Returns:
            Tuple of (final_states, swap_diagnostics).

        Raises:
            ConfigurationError: If len(states) != len(t_diffs).
        """
        raise NotSupportedError(
            "ReplicaExchangeDriver is experimental and disabled in the public release. "
            "Use ggpa.systems.phi4.LatticeRERunner or "
            "ggpa.systems.alanine_dipeptide.AlanineReplicaExchange instead."
        )

    def _attempt_swaps(self, states: List[State]) -> dict:
        """Attempt pairwise swaps between adjacent replicas.

        Uses the Metropolis-Hastings criterion based on reduced potentials.
        The intractable diffusion prior cancels in the acceptance ratio.

        Args:
            states: Current list of replica states (modified in-place on swap).

        Returns:
            Dictionary with 'swap_attempts' list of per-pair results.
        """
        swap_attempts = []
        for i in range(len(states) - 1):
            t_diff_i = self.t_diffs[i]
            t_diff_j = self.t_diffs[i + 1]
            state_i = states[i]
            state_j = states[i + 1]

            u_i_i = self.kernel.reduced_potential(state_i, t_diff_i).u_t_diff
            u_i_j = self.kernel.reduced_potential(state_j, t_diff_i).u_t_diff
            u_j_i = self.kernel.reduced_potential(state_i, t_diff_j).u_t_diff
            u_j_j = self.kernel.reduced_potential(state_j, t_diff_j).u_t_diff

            log_alpha = -(u_i_j + u_j_i) + (u_i_i + u_j_j)
            accept = log_alpha >= 0.0
            if not accept:
                accept = np.log(self.rng.uniform()) < log_alpha
            if accept:
                states[i], states[i + 1] = states[i + 1], states[i]
            swap_attempts.append({"pair": (i, i + 1), "accepted": accept, "log_alpha": float(log_alpha)})
        return {"swap_attempts": swap_attempts}
