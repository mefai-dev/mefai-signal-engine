"""
Mefai Signal Engine - Training Requirements Guide

Calculates training time estimates, data requirements, memory usage,
and generates a recommended training schedule for all ML models.
"""

import logging
import platform
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelRequirements:
    name: str
    min_candles: int
    recommended_candles: int
    approx_time_1h_data: str
    approx_time_5m_data: str
    gpu_memory_mb: int
    cpu_memory_mb: int
    training_time_cpu_minutes: float
    training_time_gpu_minutes: float
    notes: str


MODEL_SPECS = {
    "xgboost": ModelRequirements(
        name="XGBoost Signal Classifier",
        min_candles=500,
        recommended_candles=2000,
        approx_time_1h_data="~21 days",
        approx_time_5m_data="~3.5 days",
        gpu_memory_mb=0,
        cpu_memory_mb=512,
        training_time_cpu_minutes=2.0,
        training_time_gpu_minutes=2.0,
        notes="CPU-only model. Fast training, no GPU needed. "
              "Works well with limited data.",
    ),
    "transformer": ModelRequirements(
        name="Transformer Price Predictor",
        min_candles=5000,
        recommended_candles=20000,
        approx_time_1h_data="~208 days (8.7 months)",
        approx_time_5m_data="~35 days",
        gpu_memory_mb=2048,
        cpu_memory_mb=4096,
        training_time_cpu_minutes=120.0,
        training_time_gpu_minutes=15.0,
        notes="Benefits significantly from GPU acceleration. "
              "Requires the most data. Consider using 5m or 15m candles "
              "to accumulate data faster.",
    ),
    "hmm": ModelRequirements(
        name="HMM Regime Detector",
        min_candles=1000,
        recommended_candles=5000,
        approx_time_1h_data="~42 days",
        approx_time_5m_data="~7 days",
        gpu_memory_mb=0,
        cpu_memory_mb=256,
        training_time_cpu_minutes=1.0,
        training_time_gpu_minutes=1.0,
        notes="CPU-only (hmmlearn). Very fast to train. "
              "Benefits from diverse market conditions in training data.",
    ),
    "rl_position_sizer": ModelRequirements(
        name="RL Position Sizer (DQN)",
        min_candles=0,
        recommended_candles=0,
        approx_time_1h_data="N/A (needs trade history)",
        approx_time_5m_data="N/A (needs trade history)",
        gpu_memory_mb=512,
        cpu_memory_mb=1024,
        training_time_cpu_minutes=30.0,
        training_time_gpu_minutes=5.0,
        notes="Requires 10,000+ state transitions from live trading "
              "or backtesting. Approx 2 weeks of live trading data or "
              "run a backtest to generate synthetic transitions.",
    ),
    "monte_carlo": ModelRequirements(
        name="Monte Carlo Simulator",
        min_candles=100,
        recommended_candles=500,
        approx_time_1h_data="~4 days",
        approx_time_5m_data="~1 day",
        gpu_memory_mb=0,
        cpu_memory_mb=2048,
        training_time_cpu_minutes=0.5,
        training_time_gpu_minutes=0.5,
        notes="No training required - runs simulations on demand. "
              "Memory usage scales with n_simulations (10K default). "
              "CPU-only, but benefits from fast numpy (MKL/OpenBLAS).",
    ),
    "ensemble": ModelRequirements(
        name="Meta-Ensemble",
        min_candles=0,
        recommended_candles=0,
        approx_time_1h_data="N/A (calibrates online)",
        approx_time_5m_data="N/A (calibrates online)",
        gpu_memory_mb=0,
        cpu_memory_mb=128,
        training_time_cpu_minutes=0.0,
        training_time_gpu_minutes=0.0,
        notes="No offline training needed. Calibrates weights and "
              "Platt scaling online as predictions are recorded. "
              "Needs 50+ recorded outcomes for reliable calibration.",
    ),
}


class TrainingGuide:
    """Training requirements calculator and schedule generator."""

    def __init__(self):
        self.cuda_available = self._check_cuda()
        self.gpu_name = self._get_gpu_name()
        self.system_info = self._get_system_info()

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def _get_gpu_name() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except (ImportError, RuntimeError):
            pass
        return "No GPU detected"

    @staticmethod
    def _get_system_info() -> Dict:
        import os
        return {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count() or 1,
            "python_version": platform.python_version(),
        }

    def estimate_training_time(
        self, model_name: str, data_size: int
    ) -> Dict:
        """
        Estimate training time for a specific model.

        Args:
            model_name: One of the MODEL_SPECS keys
            data_size: Number of candles available

        Returns:
            Dict with time estimates and readiness status
        """
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}. Valid: {list(MODEL_SPECS.keys())}")

        spec = MODEL_SPECS[model_name]

        if spec.min_candles > 0:
            data_ready = data_size >= spec.min_candles
            data_ratio = data_size / spec.min_candles if spec.min_candles > 0 else 1.0
        else:
            data_ready = True
            data_ratio = 1.0

        # Scale training time with data size
        if spec.recommended_candles > 0:
            scale_factor = max(0.5, min(3.0, data_size / spec.recommended_candles))
        else:
            scale_factor = 1.0

        if self.cuda_available and spec.gpu_memory_mb > 0:
            estimated_minutes = spec.training_time_gpu_minutes * scale_factor
            compute_device = "GPU"
        else:
            estimated_minutes = spec.training_time_cpu_minutes * scale_factor
            compute_device = "CPU"

        return {
            "model": spec.name,
            "data_available": data_size,
            "data_minimum": spec.min_candles,
            "data_recommended": spec.recommended_candles,
            "data_ready": data_ready,
            "data_ratio": round(data_ratio, 2),
            "estimated_minutes": round(estimated_minutes, 1),
            "compute_device": compute_device,
            "memory_required_mb": (
                spec.gpu_memory_mb if self.cuda_available and spec.gpu_memory_mb > 0
                else spec.cpu_memory_mb
            ),
            "notes": spec.notes,
        }

    def full_requirements(self) -> Dict:
        """Get requirements for all models."""
        requirements = {}
        for name, spec in MODEL_SPECS.items():
            requirements[name] = {
                "name": spec.name,
                "min_candles": spec.min_candles,
                "recommended_candles": spec.recommended_candles,
                "time_to_collect_1h": spec.approx_time_1h_data,
                "time_to_collect_5m": spec.approx_time_5m_data,
                "gpu_memory_mb": spec.gpu_memory_mb,
                "cpu_memory_mb": spec.cpu_memory_mb,
                "training_time_cpu": f"{spec.training_time_cpu_minutes:.0f} min",
                "training_time_gpu": f"{spec.training_time_gpu_minutes:.0f} min",
                "notes": spec.notes,
            }
        return requirements

    def generate_training_plan(
        self,
        available_data: Dict[str, int],
        symbols: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate a recommended training plan.

        Args:
            available_data: Dict of model_name -> candles available
                OR a single int applied to all candle-based models
            symbols: List of trading symbols (for context)

        Returns:
            Formatted training plan
        """
        plan = {
            "system": {
                "gpu_available": self.cuda_available,
                "gpu_name": self.gpu_name,
                "system_info": self.system_info,
            },
            "symbols": symbols or ["not specified"],
            "models": {},
            "training_order": [],
            "total_estimated_time_minutes": 0.0,
            "recommendations": [],
        }

        ordered_models = ["hmm", "xgboost", "transformer", "rl_position_sizer", "monte_carlo", "ensemble"]

        total_time = 0.0
        ready_models = []
        not_ready_models = []

        for model_name in ordered_models:
            data_count = available_data.get(model_name, 0)
            estimate = self.estimate_training_time(model_name, data_count)
            plan["models"][model_name] = estimate

            if estimate["data_ready"]:
                ready_models.append(model_name)
                total_time += estimate["estimated_minutes"]
            else:
                not_ready_models.append(model_name)

        plan["training_order"] = ready_models
        plan["total_estimated_time_minutes"] = round(total_time, 1)

        # Generate recommendations
        if not self.cuda_available:
            plan["recommendations"].append(
                "No GPU detected. Transformer training will be slow (~2 hours on CPU). "
                "Consider using a GPU instance for transformer training."
            )

        if "transformer" in not_ready_models:
            spec = MODEL_SPECS["transformer"]
            data_count = available_data.get("transformer", 0)
            needed = spec.min_candles - data_count
            if needed > 0:
                plan["recommendations"].append(
                    f"Transformer needs {needed} more candles. "
                    f"At 1h timeframe, that is ~{needed // 24} more days of data collection. "
                    f"Tip: use 5m candles to reach minimum 12x faster."
                )

        if ready_models:
            plan["recommendations"].append(
                f"Ready to train: {', '.join(ready_models)}. "
                f"Recommended order: HMM first (fast, provides regime context), "
                f"then XGBoost (fast, baseline signals), then Transformer (slow, best accuracy)."
            )

        plan["recommendations"].append(
            "After all models are trained, the Meta-Ensemble will auto-calibrate "
            "as predictions are recorded. Allow 50+ predictions for reliable calibration."
        )

        plan["recommendations"].append(
            "RL Position Sizer requires trade history (not candles). "
            "Run a backtest with walk-forward optimization to generate training data, "
            "or accumulate 2+ weeks of live trading history."
        )

        return plan

    def print_plan(self, plan: Dict) -> str:
        """Format training plan as readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("MEFAI SIGNAL ENGINE - TRAINING PLAN")
        lines.append("=" * 60)
        lines.append("")

        sys_info = plan["system"]
        lines.append(f"GPU: {'Yes - ' + sys_info['gpu_name'] if sys_info['gpu_available'] else 'No'}")
        lines.append(f"Symbols: {', '.join(plan['symbols'])}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("MODEL STATUS")
        lines.append("-" * 60)

        for model_name, info in plan["models"].items():
            status = "READY" if info["data_ready"] else "NEEDS DATA"
            lines.append(f"\n  {info['model']} [{status}]")
            lines.append(f"    Data: {info['data_available']}/{info['data_minimum']} candles (min)")
            lines.append(f"    Device: {info['compute_device']}")
            lines.append(f"    Est. time: {info['estimated_minutes']} min")
            lines.append(f"    Memory: {info['memory_required_mb']} MB")

        lines.append("")
        lines.append("-" * 60)
        lines.append("TRAINING ORDER")
        lines.append("-" * 60)
        for i, model in enumerate(plan["training_order"], 1):
            lines.append(f"  {i}. {plan['models'][model]['model']}")

        lines.append(f"\n  Total estimated time: {plan['total_estimated_time_minutes']} min")

        lines.append("")
        lines.append("-" * 60)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 60)
        for rec in plan["recommendations"]:
            lines.append(f"  - {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)
