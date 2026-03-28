"""High-level drivers that wrap FixedTauKernel."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np

from ggpa.core.kernel import FixedTauKernel
from ggpa.core.state import State
from ggpa.core.errors import ConfigurationError
from ggpa.utils.utils import rng_from_seed


@dataclass
class FixedTauSampler:
    """Run a fixed-tau kernel for a number of steps.

    Attributes:
        kernel: The FixedTauKernel to use.
        tau: The fixed diffusion time.
    """

    kernel: FixedTauKernel
    tau: float

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
            current, diag = self.kernel.step(current, self.tau)
            diagnostics.append(diag)
        return current, diagnostics


@dataclass
class AnnealingDriver:
    """Run a kernel with a provided tau schedule.

    Attributes:
        kernel: The FixedTauKernel to use.
    """

    kernel: FixedTauKernel

    def run(self, state: State, schedule: Iterable[float]) -> Tuple[State, List]:
        """Run the kernel with an annealing schedule.

        Args:
            state: Initial state.
            schedule: Iterable of tau values (e.g., from linear_schedule).

        Returns:
            Tuple of (final_state, list_of_diagnostics).
        """
        diagnostics = []
        current = state
        for tau in schedule:
            current, diag = self.kernel.step(current, float(tau))
            diagnostics.append(diag)
        return current, diagnostics


@dataclass
class ReplicaExchangeDriver:
    """Replica exchange driver using reduced potentials for swaps.

    Runs multiple replicas at different tau values in parallel and
    periodically attempts swaps between adjacent replicas using a
    Metropolis-Hastings criterion.

    Attributes:
        kernel: The FixedTauKernel shared by all replicas.
        taus: List of tau values, one per replica (ascending order recommended).
        swap_interval: Attempt swaps every this many blocks.
        master_seed: Optional seed for reproducible swap decisions.
        rng: NumPy random generator (created from master_seed if None).
    """

    kernel: FixedTauKernel
    taus: List[float]
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
            ConfigurationError: If len(states) != len(taus).
        """
        if len(states) != len(self.taus):
            raise ConfigurationError("states and taus must have same length")
        if self.rng is None:
            self.rng = rng_from_seed(self.master_seed)
        diagnostics = []
        for block in range(n_blocks):
            for i, tau in enumerate(self.taus):
                current = states[i]
                for _ in range(inner_steps):
                    current, _ = self.kernel.step(current, tau)
                states[i] = current

            if (block + 1) % self.swap_interval == 0:
                diag = self._attempt_swaps(states)
                diagnostics.append(diag)
        return states, diagnostics

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
            tau_i = self.taus[i]
            tau_j = self.taus[i + 1]
            state_i = states[i]
            state_j = states[i + 1]

            u_i_i = self.kernel.reduced_potential(state_i, tau_i).u_tau
            u_i_j = self.kernel.reduced_potential(state_j, tau_i).u_tau
            u_j_i = self.kernel.reduced_potential(state_i, tau_j).u_tau
            u_j_j = self.kernel.reduced_potential(state_j, tau_j).u_tau

            log_alpha = -(u_i_j + u_j_i) + (u_i_i + u_j_j)
            accept = log_alpha >= 0.0
            if not accept:
                accept = np.log(self.rng.uniform()) < log_alpha
            if accept:
                states[i], states[i + 1] = states[i + 1], states[i]
            swap_attempts.append({"pair": (i, i + 1), "accepted": accept, "log_alpha": float(log_alpha)})
        return {"swap_attempts": swap_attempts}
