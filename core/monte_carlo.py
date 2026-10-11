"""
Monte Carlo probability engine.
Estimates true win probability p by sampling from a generative model of outcomes.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class MonteCarloResult:
    p: float                    # empirical win probability
    samples: np.ndarray         # raw samples (margin, score, etc.)
    ci_low: float               # 95% CI lower
    ci_high: float              # 95% CI upper
    n_sims: int


def binary_monte_carlo(
    true_logit: float,
    n_sims: int = 50_000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Simple Bernoulli Monte Carlo around a logit (log-odds) estimate.
    Useful when you already have a point estimate of strength.
    """
    rng = np.random.default_rng(seed)
    p_point = 1.0 / (1.0 + np.exp(-true_logit))
    # Add modest uncertainty around the point estimate
    logits = rng.normal(true_logit, 0.15, size=n_sims)
    probs = 1.0 / (1.0 + np.exp(-logits))
    outcomes = rng.random(n_sims) < probs
    p = float(np.mean(outcomes))
    # Wilson or normal approx CI
    se = np.sqrt(p * (1 - p) / n_sims)
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)
    return MonteCarloResult(p=p, samples=outcomes.astype(float), ci_low=ci_low, ci_high=ci_high, n_sims=n_sims)


def score_diff_monte_carlo(
    mean_margin: float,
    std_margin: float,
    n_sims: int = 50_000,
    seed: Optional[int] = None,
    win_threshold: float = 0.0,
) -> MonteCarloResult:
    """
    Simulate final score margin ~ Normal(mean, std).
    Win if margin > win_threshold (0 for moneyline, or adjusted for spread).
    """
    rng = np.random.default_rng(seed)
    margins = rng.normal(mean_margin, std_margin, size=n_sims)
    outcomes = (margins > win_threshold).astype(float)
    p = float(np.mean(outcomes))
    se = np.sqrt(p * (1 - p) / n_sims)
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)
    return MonteCarloResult(p=p, samples=margins, ci_low=ci_low, ci_high=ci_high, n_sims=n_sims)


def poisson_total_monte_carlo(
    lambda_home: float,
    lambda_away: float,
    n_sims: int = 50_000,
    seed: Optional[int] = None,
    total_line: float = 0.0,
    over: bool = True,
) -> MonteCarloResult:
    """
    Independent Poisson scoring model for totals (common in soccer/hockey, adaptable).
    """
    rng = np.random.default_rng(seed)
    home = rng.poisson(lambda_home, size=n_sims)
    away = rng.poisson(lambda_away, size=n_sims)
    totals = home + away
    if over:
        outcomes = (totals > total_line).astype(float)
    else:
        outcomes = (totals < total_line).astype(float)
    p = float(np.mean(outcomes))
    se = np.sqrt(p * (1 - p) / n_sims)
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)
    return MonteCarloResult(p=p, samples=totals.astype(float), ci_low=ci_low, ci_high=ci_high, n_sims=n_sims)


def correlated_joint_monte_carlo(
    means: np.ndarray,
    cov: np.ndarray,
    win_fn: Callable[[np.ndarray], np.ndarray],
    n_sims: int = 50_000,
    seed: Optional[int] = None,
) -> MonteCarloResult:
    """
    Multivariate normal joint simulation for correlated legs / same-game parlays.
    win_fn takes (n_sims, n_dims) array and returns boolean win vector.
    """
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(means, cov, size=n_sims)
    outcomes = win_fn(samples).astype(float)
    p = float(np.mean(outcomes))
    se = np.sqrt(p * (1 - p) / n_sims)
    ci_low = max(0.0, p - 1.96 * se)
    ci_high = min(1.0, p + 1.96 * se)
    return MonteCarloResult(p=p, samples=samples, ci_low=ci_low, ci_high=ci_high, n_sims=n_sims)
