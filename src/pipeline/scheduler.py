"""
Task Scheduler - APScheduler-based task runner.

Schedule:
- Every 1 minute: price update + quick signal check
- Every 5 minutes: full signal generation for all symbols
- Every 6 hours: model retraining
- Every 24 hours: model performance evaluation + sentiment feed refresh

Uses APScheduler's AsyncIOScheduler for non-blocking execution.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from src.pipeline.processor import SignalProcessor
from src.signals.sentiment import SentimentAnalyzer
from src.ml.trainer import Trainer

logger = logging.getLogger(__name__)


class EngineScheduler:
    """
    Manages all recurring tasks for the signal engine.

    Task hierarchy:
    1. Price monitoring (1m) - lightweight, just fetches latest price
    2. Signal generation (5m) - full multi-layer signal composition
    3. Sentiment update (5m) - RSS feed parsing
    4. Model retraining (6h) - XGBoost model training on fresh data
    5. Performance evaluation (24h) - model accuracy assessment
    """

    def __init__(
        self,
        processor: SignalProcessor,
        trainer: Trainer,
        sentiment: SentimentAnalyzer,
        config: Optional[Dict] = None,
    ):
        self.processor = processor
        self.trainer = trainer
        self.sentiment = sentiment
        self.config = config or {}

        scheduler_config = self.config.get("scheduler", {})
        self.price_interval = scheduler_config.get("price_update_seconds", 60)
        self.signal_interval = scheduler_config.get("signal_generation_seconds", 300)
        self.retrain_hours = scheduler_config.get("retrain_hours", 6)
        self.eval_hours = scheduler_config.get("evaluation_hours", 24)

        self._scheduler = AsyncIOScheduler(
            job_defaults={
                "coalesce": True,
                "max_instances": 1,
                "misfire_grace_time": 30,
            }
        )
        self._task_stats: Dict[str, Dict] = {}
        self._running = False

    async def start(self):
        """Start all scheduled tasks."""
        logger.info("Starting engine scheduler...")

        # Initial warm-up
        logger.info("Running initial data warm-up...")
        await self.processor.warm_up(limit=500)

        # Initial sentiment fetch
        logger.info("Running initial sentiment feed update...")
        await self._run_sentiment_update()

        # Initial signal generation
        logger.info("Running initial signal generation...")
        await self._run_signal_generation()

        # Schedule recurring tasks
        self._scheduler.add_job(
            self._run_price_check,
            IntervalTrigger(seconds=self.price_interval),
            id="price_check",
            name="Price Monitor",
        )

        self._scheduler.add_job(
            self._run_signal_generation,
            IntervalTrigger(seconds=self.signal_interval),
            id="signal_generation",
            name="Signal Generation",
        )

        self._scheduler.add_job(
            self._run_sentiment_update,
            IntervalTrigger(seconds=300),
            id="sentiment_update",
            name="Sentiment Update",
        )

        self._scheduler.add_job(
            self._run_model_retraining,
            IntervalTrigger(hours=self.retrain_hours),
            id="model_retrain",
            name="Model Retraining",
        )

        self._scheduler.add_job(
            self._run_performance_evaluation,
            CronTrigger(hour=0, minute=0),  # Daily at midnight UTC
            id="performance_eval",
            name="Performance Evaluation",
        )

        self._scheduler.start()
        self._running = True
        logger.info(
            "Scheduler started: price=%ds, signals=%ds, retrain=%dh",
            self.price_interval, self.signal_interval, self.retrain_hours,
        )

    async def stop(self):
        """Stop all scheduled tasks."""
        self._running = False
        self._scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    async def _run_price_check(self):
        """Quick price update for all symbols."""
        task_name = "price_check"
        start = datetime.now(timezone.utc)

        try:
            for symbol in self.processor.symbols:
                await self.processor.quick_price_check(symbol)
                await asyncio.sleep(0.05)

            self._record_task(task_name, start, success=True)
        except Exception as e:
            logger.error("Price check failed: %s", e)
            self._record_task(task_name, start, success=False, error=str(e))

    async def _run_signal_generation(self):
        """Full signal generation cycle."""
        task_name = "signal_generation"
        start = datetime.now(timezone.utc)

        try:
            results = await self.processor.process_all_symbols()
            signal_count = len(results)

            # Log summary
            for symbol, signal in results.items():
                logger.info(
                    "Signal: %s %s (confidence=%.1f%%, score=%.1f)",
                    symbol,
                    signal["direction"],
                    signal["confidence"],
                    signal["composite_score"],
                )

            self._record_task(task_name, start, success=True, details={"signals": signal_count})
        except Exception as e:
            logger.error("Signal generation failed: %s", e, exc_info=True)
            self._record_task(task_name, start, success=False, error=str(e))

    async def _run_sentiment_update(self):
        """Update sentiment from RSS feeds."""
        task_name = "sentiment_update"
        start = datetime.now(timezone.utc)

        try:
            # feedparser is synchronous, run in executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.sentiment.update_feeds)

            status = self.sentiment.get_status()
            self._record_task(task_name, start, success=True, details=status)
        except Exception as e:
            logger.error("Sentiment update failed: %s", e)
            self._record_task(task_name, start, success=False, error=str(e))

    async def _run_model_retraining(self):
        """Retrain all XGBoost models with fresh data."""
        task_name = "model_retrain"
        start = datetime.now(timezone.utc)

        try:
            logger.info("Starting model retraining cycle...")
            results = await self.trainer.train_all(interval="1h")

            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "error" not in metrics:
                    logger.info(
                        "Model %s retrained: accuracy=%.3f",
                        model_name,
                        metrics.get("final_accuracy", 0),
                    )
                elif isinstance(metrics, dict):
                    logger.warning("Model %s training issue: %s", model_name, metrics.get("error"))

            self._record_task(task_name, start, success=True, details=results)
        except Exception as e:
            logger.error("Model retraining failed: %s", e, exc_info=True)
            self._record_task(task_name, start, success=False, error=str(e))

    async def _run_performance_evaluation(self):
        """Daily model performance evaluation."""
        task_name = "performance_eval"
        start = datetime.now(timezone.utc)

        try:
            # Get model status
            model_status = self.trainer.get_all_status()

            # Get signal accuracy from history
            history = self.processor.composer.get_signal_history(limit=100)

            eval_result = {
                "models": model_status,
                "signal_count_24h": len(history),
                "timestamp": start.isoformat(),
            }

            logger.info("Performance evaluation complete: %d signals in history", len(history))
            self._record_task(task_name, start, success=True, details=eval_result)
        except Exception as e:
            logger.error("Performance evaluation failed: %s", e)
            self._record_task(task_name, start, success=False, error=str(e))

    def _record_task(
        self, task_name: str, start: datetime,
        success: bool, error: str = None, details: Dict = None
    ):
        """Record task execution statistics."""
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        stats = self._task_stats.get(task_name, {
            "total_runs": 0,
            "successes": 0,
            "failures": 0,
        })

        stats["total_runs"] += 1
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        stats["last_run"] = start.isoformat()
        stats["last_duration_seconds"] = elapsed
        stats["last_success"] = success
        if error:
            stats["last_error"] = error
        if details:
            stats["last_details"] = details

        self._task_stats[task_name] = stats

    def get_status(self) -> Dict:
        """Return scheduler status."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            })

        return {
            "running": self._running,
            "jobs": jobs,
            "task_stats": self._task_stats,
        }
