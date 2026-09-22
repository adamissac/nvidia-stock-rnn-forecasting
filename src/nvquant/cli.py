"""``nvquant`` command line interface. Each command is one pipeline stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nvquant.config import Config, config_hash, load_config
from nvquant.experiments import pipeline
from nvquant.experiments.repro import available_accelerators, git_sha, seed_everything
from nvquant.logging_utils import configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Profile YAML file.")]
DEFAULT_CONFIG = Path("configs/full.yaml")


def _setup(config: Path) -> Config:
    cfg = load_config(config)
    configure_logging(log_file=Path("logs") / f"nvquant-{cfg.profile}.log")
    seed_everything(cfg.seed)
    return cfg


@app.command()
def info(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Print the resolved config hash, git SHA, and accelerators."""
    cfg = load_config(config)
    typer.echo(
        json.dumps(
            {
                "profile": cfg.profile,
                "config_hash": config_hash(cfg),
                "git_sha": git_sha(),
                "device": cfg.device,
                "accelerators": available_accelerators(),
                "enabled_models": list(cfg.enabled_models()),
            },
            indent=2,
        )
    )


@app.command()
def data(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Download (or generate) raw data and write the manifest."""
    pipeline.stage_data(_setup(config))


@app.command("data-report")
def data_report(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Write the data-quality report."""
    pipeline.stage_data_report(_setup(config))


@app.command()
def features(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Build the feature and label store and the feature docs."""
    pipeline.stage_features(_setup(config))


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    only: Annotated[list[str] | None, typer.Option(help="Train only these models.")] = None,
) -> None:
    """Walk-forward forecasts for every enabled model, plus vol and regime models."""
    pipeline.stage_train(_setup(config), only=only)


@app.command()
def backtest(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Run strategies and benchmarks and register every strategy trial."""
    pipeline.stage_backtest(_setup(config))


@app.command()
def evaluate(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Forecast tests, significance, risk, attribution, peers, and importance."""
    pipeline.stage_evaluate(_setup(config))


@app.command()
def report(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Write results.json, the tear sheet, and the README results block."""
    pipeline.stage_report(_setup(config))


@app.command()
def reproduce(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Run every development stage from data through report."""
    pipeline.reproduce(_setup(config))


@app.command()
def lockbox(
    config: ConfigOpt = DEFAULT_CONFIG,
    force: Annotated[bool, typer.Option(help="Rerun even though a sentinel exists.")] = False,
    reason: Annotated[str | None, typer.Option(help="Required with --force.")] = None,
) -> None:
    """Evaluate the preregistered configurations on the lockbox, exactly once."""
    pipeline.stage_lockbox(_setup(config), force=force, reason=reason)


if __name__ == "__main__":
    app()
