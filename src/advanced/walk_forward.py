"""
Mefai Signal Engine - Walk-Forward Optimization

Anchored and rolling window walk-forward analysis with overfitting detection,
parameter stability analysis, and out-of-sample degradation measurement.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class WindowMetrics:
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    n_trades: int
    start_idx: int
    end_idx: int


@dataclass
class WalkForwardResult:
    windows: List[Dict]
    aggregate: Dict
    is_overfit: bool
    overfit_ratio: float
    parameter_stability: Dict
    statistical_significance: Dict
    recommendation: str


class WalkForwardOptimizer:
    """
    Walk-forward optimization and backtesting engine.

    Supports anchored (expanding) and rolling window modes.
    Detects overfitting by comparing in-sample vs out-of-sample performance.
    """

    def __init__(
        self,
        train_window: int = 30 * 24,
        test_window: int = 7 * 24,
        step_size: int = 7 * 24,
        mode: str = "rolling",
        risk_free_rate: float = 0.0,
    ):
        """
        Args:
            train_window: Training window size in candles (default 30 days of hourly)
            test_window: Testing window size in candles (default 7 days of hourly)
            step_size: Step forward size in candles (default 7 days of hourly)
            mode: "rolling" or "anchored"
            risk_free_rate: Risk-free rate per candle period
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.mode = mode
        self.risk_free_rate = risk_free_rate

    @staticmethod
    def compute_returns(prices: np.ndarray) -> np.ndarray:
        """Compute simple returns from price series."""
        return np.diff(prices) / prices[:-1]

    def compute_metrics(
        self, returns: np.ndarray, start_idx: int = 0, end_idx: int = 0
    ) -> WindowMetrics:
        """
        Compute performance metrics for a return series.

        Args:
            returns: Array of trade returns
            start_idx: Window start index
            end_idx: Window end index

        Returns:
            WindowMetrics dataclass
        """
        if len(returns) == 0:
            return WindowMetrics(
                sharpe=0.0, sortino=0.0, calmar=0.0, max_drawdown=0.0,
                win_rate=0.0, profit_factor=0.0, total_return=0.0,
                n_trades=0, start_idx=start_idx, end_idx=end_idx,
            )

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        # Sharpe ratio
        sharpe = 0.0
        if std_ret > 1e-10:
            sharpe = float((mean_ret - self.risk_free_rate) / std_ret * np.sqrt(252 * 24))

        # Sortino ratio (downside deviation)
        downside = returns[returns < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1e-10
        sortino = float((mean_ret - self.risk_free_rate) / max(downside_std, 1e-10) * np.sqrt(252 * 24))

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Calmar ratio
        annual_return = mean_ret * 252 * 24
        calmar = float(annual_return / max(max_dd, 1e-10))

        # Win rate
        wins = np.sum(returns > 0)
        total = len(returns)
        win_rate = float(wins / total) if total > 0 else 0.0

        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        profit_factor = float(gross_profit / max(gross_loss, 1e-10))

        total_return = float(np.prod(1 + returns) - 1)

        return WindowMetrics(
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            n_trades=total,
            start_idx=start_idx,
            end_idx=end_idx,
        )

    def run(
        self,
        prices: np.ndarray,
        strategy_fn: Callable[[np.ndarray, Optional[Dict]], Tuple[np.ndarray, Dict]],
        initial_params: Optional[Dict] = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization.

        Args:
            prices: Full price series
            strategy_fn: Function(prices, params) -> (returns, optimized_params)
                Takes a price window and optional parameters, returns trade returns
                and the optimized parameters for that window.
            initial_params: Starting parameters for the strategy

        Returns:
            WalkForwardResult with per-window and aggregate metrics
        """
        total_len = len(prices)
        min_required = self.train_window + self.test_window
        if total_len < min_required:
            raise ValueError(
                f"Need at least {min_required} price points, got {total_len}"
            )

        windows = []
        all_is_metrics = []
        all_oos_metrics = []
        all_oos_returns = []
        param_history = []
        current_params = initial_params

        idx = 0
        window_num = 0

        while idx + self.train_window + self.test_window <= total_len:
            if self.mode == "anchored":
                train_start = 0
            else:
                train_start = idx

            train_end = idx + self.train_window
            test_start = train_end
            test_end = min(test_start + self.test_window, total_len)

            train_prices = prices[train_start:train_end]
            test_prices = prices[test_start:test_end]

            # In-sample: optimize on training window
            is_returns, optimized_params = strategy_fn(train_prices, current_params)
            is_metrics = self.compute_metrics(is_returns, train_start, train_end)

            # Out-of-sample: apply optimized params to test window
            oos_returns, _ = strategy_fn(test_prices, optimized_params)
            oos_metrics = self.compute_metrics(oos_returns, test_start, test_end)

            window_data = {
                "window": window_num,
                "train_range": (train_start, train_end),
                "test_range": (test_start, test_end),
                "in_sample": {
                    "sharpe": is_metrics.sharpe,
                    "sortino": is_metrics.sortino,
                    "calmar": is_metrics.calmar,
                    "max_drawdown": is_metrics.max_drawdown,
                    "win_rate": is_metrics.win_rate,
                    "profit_factor": is_metrics.profit_factor,
                    "total_return": is_metrics.total_return,
                    "n_trades": is_metrics.n_trades,
                },
                "out_of_sample": {
                    "sharpe": oos_metrics.sharpe,
                    "sortino": oos_metrics.sortino,
                    "calmar": oos_metrics.calmar,
                    "max_drawdown": oos_metrics.max_drawdown,
                    "win_rate": oos_metrics.win_rate,
                    "profit_factor": oos_metrics.profit_factor,
                    "total_return": oos_metrics.total_return,
                    "n_trades": oos_metrics.n_trades,
                },
                "params": optimized_params,
            }

            windows.append(window_data)
            all_is_metrics.append(is_metrics)
            all_oos_metrics.append(oos_metrics)
            all_oos_returns.extend(oos_returns.tolist() if len(oos_returns) > 0 else [])
            param_history.append(optimized_params)

            current_params = optimized_params
            idx += self.step_size
            window_num += 1

        if not windows:
            raise ValueError("No valid windows could be created")

        # Aggregate metrics
        combined_oos = np.array(all_oos_returns) if all_oos_returns else np.array([0.0])
        aggregate_metrics = self.compute_metrics(combined_oos)

        aggregate = {
            "sharpe": aggregate_metrics.sharpe,
            "sortino": aggregate_metrics.sortino,
            "calmar": aggregate_metrics.calmar,
            "max_drawdown": aggregate_metrics.max_drawdown,
            "win_rate": aggregate_metrics.win_rate,
            "profit_factor": aggregate_metrics.profit_factor,
            "total_return": aggregate_metrics.total_return,
            "n_windows": len(windows),
            "n_trades": aggregate_metrics.n_trades,
        }

        # Overfitting detection
        is_sharpes = [m.sharpe for m in all_is_metrics]
        oos_sharpes = [m.sharpe for m in all_oos_metrics]

        avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        if avg_oos_sharpe != 0:
            overfit_ratio = float(avg_is_sharpe / avg_oos_sharpe)
        elif avg_is_sharpe > 0:
            overfit_ratio = float("inf")
        else:
            overfit_ratio = 1.0

        is_overfit = overfit_ratio > 2.0

        # Statistical significance (t-test on OOS returns)
        sig_result = self._test_significance(combined_oos)

        # Parameter stability
        param_stability = self._analyze_parameter_stability(param_history)

        # Recommendation
        recommendation = self._generate_recommendation(
            is_overfit, overfit_ratio, aggregate_metrics, sig_result
        )

        return WalkForwardResult(
            windows=windows,
            aggregate=aggregate,
            is_overfit=is_overfit,
            overfit_ratio=overfit_ratio,
            parameter_stability=param_stability,
            statistical_significance=sig_result,
            recommendation=recommendation,
        )

    @staticmethod
    def _test_significance(returns: np.ndarray) -> Dict:
        """Test if returns are statistically different from zero using t-test."""
        if len(returns) < 3:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant_95": False, "significant_99": False}

        t_stat, p_value = scipy_stats.ttest_1samp(returns, 0.0)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_95": bool(p_value < 0.05),
            "significant_99": bool(p_value < 0.01),
            "n_samples": len(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
        }

    @staticmethod
    def _analyze_parameter_stability(param_history: List[Optional[Dict]]) -> Dict:
        """
        Analyze how much optimized parameters change across windows.

        Returns coefficient of variation for each numeric parameter.
        """
        if not param_history or all(p is None for p in param_history):
            return {"stable": True, "note": "No parameters to analyze"}

        valid_params = [p for p in param_history if p is not None]
        if not valid_params:
            return {"stable": True, "note": "No valid parameters"}

        all_keys = set()
        for p in valid_params:
            all_keys.update(p.keys())

        stability = {}
        unstable_count = 0

        for key in all_keys:
            values = []
            for p in valid_params:
                v = p.get(key)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    values.append(float(v))

            if len(values) < 2:
                continue

            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)

            if abs(mean_val) > 1e-10:
                cv = float(std_val / abs(mean_val))
            else:
                cv = float(std_val)

            is_stable = cv < 0.5
            if not is_stable:
                unstable_count += 1

            stability[key] = {
                "mean": float(mean_val),
                "std": float(std_val),
                "cv": cv,
                "stable": is_stable,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        stability["_summary"] = {
            "total_params": len(stability) - 1,
            "unstable_params": unstable_count,
            "overall_stable": unstable_count == 0,
        }

        return stability

    @staticmethod
    def _generate_recommendation(
        is_overfit: bool, overfit_ratio: float,
        metrics: WindowMetrics, significance: Dict
    ) -> str:
        """Generate a text recommendation based on walk-forward results."""
        parts = []

        if is_overfit:
            parts.append(
                f"WARNING: Strategy shows signs of overfitting "
                f"(in-sample/out-of-sample ratio: {overfit_ratio:.1f}x). "
                f"Consider simplifying the model or using more regularization."
            )
        else:
            parts.append("No significant overfitting detected.")

        if not significance.get("significant_95", False):
            parts.append(
                "Out-of-sample returns are NOT statistically significant at 95% level. "
                "Results could be due to chance."
            )
        elif significance.get("significant_99", False):
            parts.append("Out-of-sample returns are statistically significant at 99% level.")
        else:
            parts.append(
                "Out-of-sample returns are statistically significant at 95% level "
                "but not at 99%."
            )

        if metrics.sharpe > 1.5:
            parts.append(f"Strong risk-adjusted performance (Sharpe: {metrics.sharpe:.2f}).")
        elif metrics.sharpe > 0.5:
            parts.append(f"Moderate risk-adjusted performance (Sharpe: {metrics.sharpe:.2f}).")
        elif metrics.sharpe > 0:
            parts.append(f"Weak positive performance (Sharpe: {metrics.sharpe:.2f}).")
        else:
            parts.append(f"Negative performance (Sharpe: {metrics.sharpe:.2f}). Do not deploy.")

        if metrics.max_drawdown > 0.20:
            parts.append(
                f"High max drawdown ({metrics.max_drawdown:.1%}). "
                f"Consider adding drawdown protection."
            )

        return " ".join(parts)

    def generate_report(self, result: WalkForwardResult) -> Dict:
        """Generate a comprehensive walk-forward report."""
        return {
            "summary": {
                "mode": self.mode,
                "train_window": self.train_window,
                "test_window": self.test_window,
                "step_size": self.step_size,
                "n_windows": len(result.windows),
                "is_overfit": result.is_overfit,
                "overfit_ratio": result.overfit_ratio,
            },
            "aggregate_metrics": result.aggregate,
            "statistical_significance": result.statistical_significance,
            "parameter_stability": result.parameter_stability,
            "recommendation": result.recommendation,
            "per_window": result.windows,
        }
