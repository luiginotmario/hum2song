"""Weights & Biases logging. Importing this module requires the wandb package."""

import os

import wandb

from hum2song.tracking import NullTracker


class WandbTracker:
    """Log scalars to a W&B run. Construct this only when WANDB_API_KEY is set."""

    def __init__(self, run_name: str, config: dict) -> None:
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "hum2song"),
            name=run_name,
            config=config,
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log one training step."""
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Close the run."""
        wandb.finish()


def build_tracker(run_name: str, config: dict):
    """Return a W&B tracker when WANDB_API_KEY is set, otherwise a no-op."""
    if not os.environ.get("WANDB_API_KEY"):
        return NullTracker()
    return WandbTracker(run_name, config)
