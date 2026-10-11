"""
Same-game and cross-game correlation estimation for parlays.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Sequence
from dataclasses import dataclass


@dataclass
class CorrelationMatrix:
    labels: List[str]
    matrix: np.ndarray          # square, symmetric, diagonal 1

    def pairwise(self, a: str, b: str) -> float:
        try:
            i, j = self.labels.index(a), self.labels.index(b)
            return float(self.matrix[i, j])
        except ValueError:
            return 0.0


def estimate_same_game_correlation(
    leg_ids: Sequence[str],
    market_types: Sequence[str],
    base_same_game: float = 0.35,
    base_same_team: float = 0.45,
    base_cross: float = 0.05,
) -> CorrelationMatrix:
    """
    Heuristic correlation matrix when historical joint data is unavailable.
    Rules of thumb (calibrate later on historical co-occurrence):
      - Same game, different markets (ML + total, ML + spread): ~0.25–0.45
      - Same team props: higher
      - Different games: near 0 (small residual from slate-wide factors)
    """
    n = len(leg_ids)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            # parse crude groups from id prefix if present (EVENT-xxx)
            gi = leg_ids[i].split("-")[0] if "-" in leg_ids[i] else leg_ids[i]
            gj = leg_ids[j].split("-")[0] if "-" in leg_ids[j] else leg_ids[j]
            mi, mj = market_types[i].lower(), market_types[j].lower()
            if gi == gj:
                if "team" in mi or "team" in mj or "player" in mi:
                    c = base_same_team
                else:
                    c = base_same_game
            else:
                c = base_cross
            # slight market-type boost
            if mi == mj:
                c = min(0.85, c + 0.05)
            mat[i, j] = mat[j, i] = c
    return CorrelationMatrix(labels=list(leg_ids), matrix=mat)


def correlation_adjusted_joint_prob(
    probs: Sequence[float],
    corr: CorrelationMatrix,
    method: str = "gaussian_copula",
    n_sims: int = 30_000,
    seed: int = 0,
) -> float:
    """
    Estimate P(all win) under a Gaussian copula with the given correlation matrix.
    Falls back to independence if matrix is identity.
    """
    p = np.asarray(probs, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0
    if n == 1:
        return float(p[0])

    from scipy.special import erfinv, erf
    # thresholds for probit
    z = np.sqrt(2) * erfinv(2 * np.clip(p, 1e-6, 1 - 1e-6) - 1)
    rng = np.random.default_rng(seed)
    # ensure PD
    mat = corr.matrix.copy()
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 1.0)
    # jitter if needed
    eig = np.linalg.eigvalsh(mat)
    if eig.min() < 1e-8:
        mat += np.eye(n) * (1e-6 - eig.min())
    samples = rng.multivariate_normal(np.zeros(n), mat, size=n_sims)
    wins = samples < z[None, :]
    joint = float(np.mean(np.all(wins, axis=1)))
    return joint


def historical_cooccurrence_matrix(
    outcomes: np.ndarray,
    labels: List[str],
) -> CorrelationMatrix:
    """
    outcomes: (n_games, n_legs) binary matrix of historical results.
    Returns Pearson correlation of the Bernoulli outcomes as a proxy.
    """
    if outcomes.shape[1] != len(labels):
        raise ValueError("shape mismatch")
    # avoid zero variance
    std = outcomes.std(axis=0)
    std[std < 1e-9] = 1.0
    centered = (outcomes - outcomes.mean(axis=0)) / std
    mat = (centered.T @ centered) / max(1, outcomes.shape[0] - 1)
    np.fill_diagonal(mat, 1.0)
    mat = np.clip(mat, -0.95, 0.95)
    return CorrelationMatrix(labels=labels, matrix=mat)
