"""
Mefai Signal Engine - Monte Carlo Risk Simulation

Geometric Brownian Motion + Merton Jump Diffusion for price path simulation.
Computes VaR, CVaR, max drawdown distributions, and portfolio-level risk
using Cholesky decomposition for correlated assets.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

N_SIMULATIONS = 10_000
TIME_HORIZONS = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
    "1w": 168,
}
STRESS_MULTIPLIERS = [2.0, 3.0, 5.0]


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    max_drawdown_mean: float
    max_drawdown_95: float
    expected_return: float
    return_std: float
    skewness: float
    kurtosis: float
    probability_of_loss: float
    probability_of_large_loss: float


@dataclass
class SimulationResult:
    risk_metrics: Dict[str, RiskMetrics]
    stress_test: Dict[str, RiskMetrics]
    final_prices: np.ndarray
    price_paths: Optional[np.ndarray]
    parameters: Dict


class MonteCarloSimulator:
    """
    Monte Carlo risk simulation engine.

    Supports:
        - Geometric Brownian Motion (GBM)
        - Merton Jump Diffusion for tail risk
        - Portfolio-level simulation with correlated assets
        - Stress testing at 2x, 3x, 5x volatility
    """

    def __init__(self, n_simulations: int = N_SIMULATIONS, seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def estimate_parameters(prices: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Estimate GBM and jump diffusion parameters from historical prices.

        Args:
            prices: Historical price array
            dt: Time step in hours (1.0 for hourly candles)

        Returns:
            Dict with drift (mu), volatility (sigma), and jump parameters
        """
        log_returns = np.diff(np.log(prices + 1e-10))
        log_returns = log_returns[np.isfinite(log_returns)]

        if len(log_returns) < 10:
            raise ValueError("Need at least 10 price points for parameter estimation")

        mu = float(np.mean(log_returns) / dt)
        sigma = float(np.std(log_returns) / np.sqrt(dt))

        excess_kurtosis = float(stats.kurtosis(log_returns))
        jump_intensity = 0.0
        jump_mean = 0.0
        jump_std = 0.0

        if excess_kurtosis > 1.0:
            threshold = np.mean(log_returns) + 2.5 * np.std(log_returns)
            jumps = log_returns[np.abs(log_returns) > threshold]

            if len(jumps) > 2:
                jump_intensity = len(jumps) / (len(log_returns) * dt)
                jump_mean = float(np.mean(jumps))
                jump_std = float(np.std(jumps))
                non_jump_returns = log_returns[np.abs(log_returns) <= threshold]
                if len(non_jump_returns) > 1:
                    sigma = float(np.std(non_jump_returns) / np.sqrt(dt))

        return {
            "mu": mu,
            "sigma": sigma,
            "jump_intensity": jump_intensity,
            "jump_mean": jump_mean,
            "jump_std": jump_std,
            "excess_kurtosis": excess_kurtosis,
            "n_observations": len(log_returns),
        }

    def simulate_gbm(
        self, s0: float, mu: float, sigma: float, n_steps: int, dt: float = 1.0
    ) -> np.ndarray:
        """
        Simulate price paths using Geometric Brownian Motion.

        dS = mu * S * dt + sigma * S * dW

        Args:
            s0: Initial price
            mu: Drift (annualized)
            sigma: Volatility (annualized)
            n_steps: Number of time steps
            dt: Time step fraction

        Returns:
            Price paths array of shape (n_simulations, n_steps + 1)
        """
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = s0

        z = self.rng.standard_normal((self.n_simulations, n_steps))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * z[:, t])

        return paths

    def simulate_jump_diffusion(
        self, s0: float, mu: float, sigma: float,
        jump_intensity: float, jump_mean: float, jump_std: float,
        n_steps: int, dt: float = 1.0
    ) -> np.ndarray:
        """
        Simulate price paths using Merton Jump Diffusion.

        dS/S = (mu - lambda * k) * dt + sigma * dW + J * dN

        where J ~ N(jump_mean, jump_std^2), dN ~ Poisson(lambda * dt)

        Args:
            s0: Initial price
            mu: Drift
            sigma: Diffusion volatility
            jump_intensity: Average number of jumps per unit time (lambda)
            jump_mean: Mean log jump size
            jump_std: Std of log jump size
            n_steps: Number of time steps
            dt: Time step fraction

        Returns:
            Price paths array of shape (n_simulations, n_steps + 1)
        """
        paths = np.zeros((self.n_simulations, n_steps + 1))
        paths[:, 0] = s0

        k = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
        compensated_drift = (mu - jump_intensity * k - 0.5 * sigma ** 2) * dt

        for t in range(n_steps):
            z = self.rng.standard_normal(self.n_simulations)
            n_jumps = self.rng.poisson(jump_intensity * dt, self.n_simulations)
            jump_sizes = np.zeros(self.n_simulations)

            for i in range(self.n_simulations):
                if n_jumps[i] > 0:
                    jumps = self.rng.normal(jump_mean, jump_std, n_jumps[i])
                    jump_sizes[i] = np.sum(jumps)

            log_return = compensated_drift + sigma * np.sqrt(dt) * z + jump_sizes
            paths[:, t + 1] = paths[:, t] * np.exp(log_return)

        return paths

    @staticmethod
    def compute_risk_metrics(paths: np.ndarray) -> RiskMetrics:
        """
        Compute risk metrics from simulated price paths.

        Args:
            paths: Price paths of shape (n_simulations, n_steps + 1)

        Returns:
            RiskMetrics dataclass
        """
        final_returns = (paths[:, -1] - paths[:, 0]) / paths[:, 0]

        var_95 = float(-np.percentile(final_returns, 5))
        var_99 = float(-np.percentile(final_returns, 1))

        tail_95 = final_returns[final_returns <= -var_95]
        cvar_95 = float(-np.mean(tail_95)) if len(tail_95) > 0 else var_95

        tail_99 = final_returns[final_returns <= -var_99]
        cvar_99 = float(-np.mean(tail_99)) if len(tail_99) > 0 else var_99

        max_drawdowns = np.zeros(paths.shape[0])
        for i in range(paths.shape[0]):
            cummax = np.maximum.accumulate(paths[i])
            drawdowns = (cummax - paths[i]) / cummax
            max_drawdowns[i] = np.max(drawdowns)

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            max_drawdown_mean=float(np.mean(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            expected_return=float(np.mean(final_returns)),
            return_std=float(np.std(final_returns)),
            skewness=float(stats.skew(final_returns)),
            kurtosis=float(stats.kurtosis(final_returns)),
            probability_of_loss=float(np.mean(final_returns < 0)),
            probability_of_large_loss=float(np.mean(final_returns < -0.05)),
        )

    def run_single_asset(
        self, prices: np.ndarray, dt: float = 1.0, store_paths: bool = False
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation for a single asset.

        Args:
            prices: Historical price array
            dt: Time step in hours
            store_paths: Whether to store full price paths (memory intensive)

        Returns:
            SimulationResult with risk metrics for all time horizons and stress tests
        """
        params = self.estimate_parameters(prices, dt)
        s0 = float(prices[-1])
        use_jumps = params["jump_intensity"] > 0

        risk_by_horizon = {}
        stress_by_horizon = {}
        all_final_prices = None

        for horizon_name, horizon_hours in TIME_HORIZONS.items():
            n_steps = max(1, int(horizon_hours / dt))

            if use_jumps:
                paths = self.simulate_jump_diffusion(
                    s0, params["mu"], params["sigma"],
                    params["jump_intensity"], params["jump_mean"], params["jump_std"],
                    n_steps, dt,
                )
            else:
                paths = self.simulate_gbm(s0, params["mu"], params["sigma"], n_steps, dt)

            risk_by_horizon[horizon_name] = self.compute_risk_metrics(paths)

            if horizon_name == "1d":
                all_final_prices = paths[:, -1]

            for mult in STRESS_MULTIPLIERS:
                stress_sigma = params["sigma"] * mult
                label = f"{horizon_name}_{mult:.0f}x_vol"

                if use_jumps:
                    stress_paths = self.simulate_jump_diffusion(
                        s0, params["mu"], stress_sigma,
                        params["jump_intensity"] * mult,
                        params["jump_mean"], params["jump_std"] * np.sqrt(mult),
                        n_steps, dt,
                    )
                else:
                    stress_paths = self.simulate_gbm(
                        s0, params["mu"], stress_sigma, n_steps, dt
                    )

                stress_by_horizon[label] = self.compute_risk_metrics(stress_paths)

        return SimulationResult(
            risk_metrics=risk_by_horizon,
            stress_test=stress_by_horizon,
            final_prices=all_final_prices if all_final_prices is not None else np.array([]),
            price_paths=None,
            parameters=params,
        )

    def simulate_portfolio(
        self, assets: Dict[str, np.ndarray], weights: Dict[str, float],
        dt: float = 1.0, horizon_hours: int = 24
    ) -> Dict:
        """
        Portfolio-level Monte Carlo with correlated assets via Cholesky decomposition.

        Args:
            assets: Dict of symbol -> price array
            weights: Dict of symbol -> portfolio weight (should sum to 1)
            dt: Time step in hours
            horizon_hours: Simulation horizon in hours

        Returns:
            Portfolio risk metrics dict
        """
        symbols = list(assets.keys())
        n_assets = len(symbols)
        n_steps = max(1, int(horizon_hours / dt))

        returns_matrix = []
        asset_params = {}
        for sym in symbols:
            prices = assets[sym]
            log_ret = np.diff(np.log(prices + 1e-10))
            log_ret = log_ret[np.isfinite(log_ret)]
            returns_matrix.append(log_ret)
            asset_params[sym] = self.estimate_parameters(prices, dt)

        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = np.array([r[-min_len:] for r in returns_matrix])

        corr_matrix = np.corrcoef(returns_matrix)

        try:
            cholesky = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            np.fill_diagonal(corr_matrix, 1.0)
            cholesky = np.linalg.cholesky(corr_matrix)

        portfolio_values = np.zeros((self.n_simulations, n_steps + 1))
        portfolio_values[:, 0] = 1.0

        w = np.array([weights.get(sym, 1.0 / n_assets) for sym in symbols])
        w = w / w.sum()

        for t in range(n_steps):
            z_uncorrelated = self.rng.standard_normal((self.n_simulations, n_assets))
            z_correlated = z_uncorrelated @ cholesky.T

            asset_returns = np.zeros((self.n_simulations, n_assets))
            for i, sym in enumerate(symbols):
                p = asset_params[sym]
                drift = (p["mu"] - 0.5 * p["sigma"] ** 2) * dt
                asset_returns[:, i] = np.exp(drift + p["sigma"] * np.sqrt(dt) * z_correlated[:, i])

            portfolio_return = np.sum(w * asset_returns, axis=1)
            portfolio_values[:, t + 1] = portfolio_values[:, t] * portfolio_return

        metrics = self.compute_risk_metrics(portfolio_values)

        return {
            "portfolio_metrics": metrics,
            "correlation_matrix": corr_matrix.tolist(),
            "asset_weights": {sym: float(w[i]) for i, sym in enumerate(symbols)},
            "n_assets": n_assets,
            "horizon_hours": horizon_hours,
            "diversification_ratio": float(self._diversification_ratio(w, returns_matrix)),
        }

    @staticmethod
    def _diversification_ratio(weights: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute diversification ratio = weighted avg vol / portfolio vol.
        A ratio > 1 indicates diversification benefit.
        """
        individual_vols = np.std(returns, axis=1)
        cov_matrix = np.cov(returns)
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(max(portfolio_var, 1e-10))
        weighted_vol_sum = np.sum(weights * individual_vols)
        return weighted_vol_sum / portfolio_vol if portfolio_vol > 0 else 1.0

    def summary(self, result: SimulationResult) -> Dict:
        """Generate a human-readable risk summary."""
        summary = {"parameters": result.parameters, "horizons": {}}

        for horizon, metrics in result.risk_metrics.items():
            summary["horizons"][horizon] = {
                "VaR_95": f"{metrics.var_95:.2%}",
                "VaR_99": f"{metrics.var_99:.2%}",
                "CVaR_95": f"{metrics.cvar_95:.2%}",
                "CVaR_99": f"{metrics.cvar_99:.2%}",
                "max_drawdown_mean": f"{metrics.max_drawdown_mean:.2%}",
                "max_drawdown_95th": f"{metrics.max_drawdown_95:.2%}",
                "expected_return": f"{metrics.expected_return:.4%}",
                "probability_of_loss": f"{metrics.probability_of_loss:.1%}",
            }

        stress_summary = {}
        for label, metrics in result.stress_test.items():
            stress_summary[label] = {
                "VaR_95": f"{metrics.var_95:.2%}",
                "CVaR_95": f"{metrics.cvar_95:.2%}",
                "max_drawdown_95th": f"{metrics.max_drawdown_95:.2%}",
            }
        summary["stress_tests"] = stress_summary

        return summary
