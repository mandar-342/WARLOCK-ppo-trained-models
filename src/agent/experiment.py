from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from src.utils import config, root


class ExperimentManager:
    """
    Manages the complete lifecycle of a single WARLOCK experiment.

    Each training run is assigned a unique timestamped directory
    containing configuration snapshots, checkpoints, TensorBoard logs,
    evaluation metrics and generated reports.

    The class is intentionally independent of any reinforcement learning
    algorithm. It only manages experiment artifacts.
    """

    def __init__(
        self,
        experiment_name: str,
        run_directory: str | Path | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._started_at = datetime.now(timezone.utc)
        self._timestamp = self._started_at.strftime("%Y%m%d_%H%M%S")
        if run_directory is None:
            # Timestamp alone is only second-resolution, so two runs
            # (e.g. concurrent Optuna worker processes, or fast-starting
            # multi-seed jobs) started in the same second would otherwise
            # collide on this directory name and clobber each other's
            # checkpoints/logs. Appending the PID and a short random
            # suffix makes it collision-safe across concurrent processes
            # while keeping the directory sortable-by-start-time.
            run_id = f"{self._timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"
            self._run_directory = root(
                "experiments",
                experiment_name,
                run_id,
            )
        else:
            self._run_directory = Path(run_directory)
        self._run_directory = self._run_directory.resolve()

        self._checkpoints_directory = self._run_directory / "checkpoints"
        self._tensorboard_directory = self._run_directory / "tensorboard"
        self._logs_directory = self._run_directory / "logs"
        self._plots_directory = self._run_directory / "plots"

        self._config_snapshot = self._run_directory / "config.yaml"
        self._metadata_file = self._run_directory / "metadata.json"

        self._model_file = self._run_directory / "model.zip"
        self._best_model_file = self._run_directory / "best_model.zip"

        self._metrics_file = self._run_directory / "metrics.json"
        self._report_file = self._run_directory / "report.pdf"

        self._create_directories()

        self.snapshot_config()

        self.save_metadata()

        logger.info(
            "Created experiment directory: {}",
            self._run_directory,
        )

    def _create_directories(self) -> None:
        """
        Creates the complete experiment directory hierarchy.
        """

        directories = (
            self._run_directory,
            self._checkpoints_directory,
            self._tensorboard_directory,
            self._logs_directory,
            self._plots_directory,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            
    @property
    def experiment_name(self) -> str:
        """
        Returns the experiment name.
        """
        return self._experiment_name

    @property
    def timestamp(self) -> str:
        """
        Returns the experiment timestamp.
        """
        return self._timestamp

    @property
    def run_directory(self) -> Path:
        """
        Returns the root directory of the current experiment.
        """
        return self._run_directory

    @property
    def checkpoints_directory(self) -> Path:
        return self._checkpoints_directory

    @property
    def tensorboard_directory(self) -> Path:
        return self._tensorboard_directory

    @property
    def logs_directory(self) -> Path:
        return self._logs_directory

    @property
    def plots_directory(self) -> Path:
        return self._plots_directory

    @property
    def model_path(self) -> Path:
        return self._model_file

    @property
    def best_model_path(self) -> Path:
        return self._best_model_file

    @property
    def metrics_path(self) -> Path:
        return self._metrics_file

    @property
    def report_path(self) -> Path:
        return self._report_file

    @property
    def metadata_path(self) -> Path:
        return self._metadata_file

    @property
    def config_snapshot_path(self) -> Path:
        return self._config_snapshot

    def _git_commit(self) -> str:
        """
        Returns the current Git commit hash.

        Returns
        -------
        str
            Git SHA or "unknown" if unavailable.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root(),
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout.strip()

        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return "unknown"

    def _metadata(self) -> dict[str, Any]:
        """
        Builds experiment metadata.
        """
        training_cfg = config.get("training", {})
        ppo_cfg = config.get("ppo", {})
        
        return {
            "experiment_name": self._experiment_name,
            "timestamp": self._started_at.isoformat(),
            "git_commit": self._git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "algorithm": "PPO",
            "policy": ppo_cfg.get("policy"),
            "device": training_cfg.get("device"),
             "seed": training_cfg.get("seed"),
            "n_envs": training_cfg.get("n_envs", 1),
            "policy_kwargs": ppo_cfg.get("policy_kwargs", {}),
        }
        
        
    def snapshot_config(self) -> None:
        """
        Creates an immutable snapshot of the configuration file used
        for this experiment.
        """

        source = root ("config.yaml")

        if not source.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {source}"
            )

        shutil.copy2(
            source,
            self._config_snapshot,
        )

    def save_metadata(
        self,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Saves experiment metadata.

        Parameters
        ----------
        extra
            Optional metadata fields to merge into the generated
            metadata.
        """

        metadata = self._metadata()

        if extra:
            metadata.update(extra)

        with self._metadata_file.open(
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                metadata,
                fp,
                indent=4,
                sort_keys=True,
            )

    def update_metadata(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Updates the metadata file while preserving existing fields.

        Parameters
        ----------
        **kwargs
            Metadata fields to update.
        """

        if self._metadata_file.exists():
            with self._metadata_file.open(
                "r",
                encoding="utf-8",
            ) as fp:
                metadata = json.load(fp)
        else:
            metadata = self._metadata()

        metadata.update(kwargs)

        with self._metadata_file.open(
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                metadata,
                fp,
                indent=4,
                sort_keys=True,
            )
            
    def artifact_path(
         self,
        name: str,
    )-> Path:
        
        
        """
    Returns a path inside the experiment directory.

    Parameters
    ----------
    name
        Relative artifact filename.

    Returns
    -------
    Path
        Artifact path.

    Raises
    ------
    ValueError
        If an absolute path is provided.
    """

        path = Path(name)

        if path.is_absolute():
            
            raise ValueError(
            "Artifact name must be a relative path."
        )

        return self._run_directory / path

    def save_metrics(
        self,
        metrics: dict[str, Any],
    ) -> None:
        """
        Saves evaluation metrics to metrics.json.

        Parameters
        ----------
        metrics
            Dictionary containing evaluation metrics.
        """

        with self._metrics_file.open(
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(
                metrics,
                fp,
                indent=4,
                sort_keys=True,
            )

    def load_metrics(self) -> dict[str, Any]:
        
        """
        Loads metrics from metrics.json.

        Returns
        -------
        dict[str, Any]
            Metrics dictionary.

        Raises
        ------
        FileNotFoundError
            If metrics.json does not exist.
        """

        if not self._metrics_file.exists():
            raise FileNotFoundError(
                f"Metrics file does not exist: {self._metrics_file}"
            )

        with self._metrics_file.open(
            "r",
            encoding="utf-8",
        ) as fp:
            return json.load(fp)

    def exists(self) -> bool:
        """
        Returns whether the experiment directory exists.
        """
        return self._run_directory.exists()

    def cleanup(
        self,
        remove_empty: bool = False,
    ) -> None:
        """
        Cleans up an experiment directory.

        Parameters
        ----------
        remove_empty
            If True, removes the experiment directory only if it is empty.
        """

        if not remove_empty:
            return

        try:
            self._run_directory.rmdir()
            logger.info(
                "Removed empty experiment directory: {}",
                self._run_directory,
            )
        except OSError as exc:
             logger.debug( "Experiment directory not removed: {}",exc,)
            
            
    def __str__(self) -> str:
        """
        Returns the experiment directory.
        """
        return str(self._run_directory)

    def __repr__(self) -> str:
        """
        Returns a developer-friendly representation.
        """
        return (
            f"{self.__class__.__name__}("
            f"experiment_name={self._experiment_name!r}, "
            f"timestamp={self._timestamp!r})"
        )
