"""Run trackers. Weights & Biases is optional and lives in wandb_tracker.py."""


class NullTracker:
    """Tracker used when WANDB_API_KEY is unset."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Ignore metrics."""
        return

    def finish(self) -> None:
        """Nothing to close."""
        return
