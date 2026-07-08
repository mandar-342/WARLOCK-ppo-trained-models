from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agent.callbacks import CallbackFactory
from src.agent.experiment import ExperimentManager
from src.env.gym_bitcoin import GymBitcoinEnv
from src.utils import config


class PPOTrainer:
    """
    High-level trainer responsible for orchestrating PPO training.

    Responsibilities
    ----------------
    - Create experiment
    - Create environments
    - Build PPO model
    - Attach callbacks
    - Train
    - Save final model
    """

    def __init__(
        self,
        total_timesteps: int | None = None,
    ) -> None:

        self._cfg = config
        self._ppo_cfg = config["ppo"]

        self._experiment = ExperimentManager(
            experiment_name="ppo_baseline",
        )

        self._training_cfg = config["training"]

        self._total_timesteps = (
            total_timesteps
            if total_timesteps is not None
            else int(self._training_cfg["total_timesteps"])
        )
        
        self._device = self._training_cfg.get("device", "auto")

        logger.info(
            "Initializing PPO trainer "
            "(device={}, timesteps={:,})",
            self._device,
            self._total_timesteps,
        )

        self._train_env = self._build_environment()
        self._eval_env = self._build_environment()

        self._callbacks = CallbackFactory(
            experiment=self._experiment,
            eval_env=self._eval_env,
        ).build()

        self._model: PPO | None = None

    def _build_environment(
        self,
        
    ) -> DummyVecEnv:
        """
        Builds a monitored vectorized environment.
        """

        def make_env() -> Monitor:
            env = GymBitcoinEnv()
            return Monitor(env)

        return DummyVecEnv([make_env])
    
    def _policy_kwargs(self) -> dict[str, Any]:
        """
        Returns the PPO policy architecture.
        """
        architecture = self._ppo_cfg["policy_kwargs"]
        return {
            "activation_fn": torch.nn.ReLU,
             "net_arch": architecture["net_arch"],
        }

    def _build_model(self) -> PPO:
        """
        Creates the PPO model.
        """

        logger.info("Building PPO model.")

        model = PPO(
            policy="MlpPolicy",
            env=self._train_env,
            learning_rate=float(self._ppo_cfg["learning_rate"]),
            n_steps=int(self._ppo_cfg["n_steps"]),
            batch_size=int(self._ppo_cfg["batch_size"]),
            n_epochs=int(self._ppo_cfg["n_epochs"]),
            gamma=float(self._ppo_cfg["gamma"]),
            gae_lambda=float(self._ppo_cfg["gae_lambda"]),
            clip_range=float(self._ppo_cfg["clip_range"]),
            ent_coef=float(self._ppo_cfg["ent_coef"]),
            vf_coef=float(self._ppo_cfg["vf_coef"]),
            max_grad_norm=float(self._ppo_cfg["max_grad_norm"]),
            seed=int(self._training_cfg["seed"]),
            device=self._device,
            tensorboard_log=str(
                self._experiment.tensorboard_directory
            ),
            policy_kwargs=self._policy_kwargs(),
            verbose=1,
        )

        logger.info("PPO model created successfully.")

        return model

    def train(self) -> PPO:
        """
        Trains the PPO agent.

        Returns
        -------
        PPO
            Trained PPO model.
        """

        logger.info("Starting PPO training.")

        self._model = self._build_model()

        self._model.learn(
            total_timesteps=self._total_timesteps,
            callback=self._callbacks,
            tb_log_name=self._experiment.experiment_name,
            progress_bar=True,
        )

        logger.info("Training completed.")

        logger.info(
            "Saving trained model to {}",
            self._experiment.model_path,
        )

        self._model.save(
            str(self._experiment.model_path)
        )

        self._experiment.update_metadata(
            total_timesteps=self._total_timesteps,
            model_path=str(self._experiment.model_path),
            best_model_path=str(
                self._experiment.best_model_path
            ),
            completed=True,
        )

        logger.success(
            "Model successfully saved."
        )

        return self._model
    
    @property
    def model(self) -> PPO:
        """
        Returns the trained PPO model.

        Raises
        ------
        RuntimeError
            If the model has not been created or loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "PPO model has not been initialized."
            )

        return self._model

    @property
    def experiment(self) -> ExperimentManager:
        """
        Returns the associated experiment manager.
        """
        return self._experiment

    def load(
        self,
        model_path: str | None = None,
    ) -> PPO:
        """
        Loads a PPO model.

        Parameters
        ----------
        model_path
            Optional path to the model. If omitted, loads the
            experiment's default model.

        Returns
        -------
        PPO
            Loaded PPO model.
        """

        path = (
            self._experiment.model_path
            if model_path is None
            else model_path
        )

        logger.info(
            "Loading PPO model from {}",
            path,
        )

        self._model = PPO.load(
            str(path),
            env=self._train_env,
            device=self._device,
        )

        logger.success("Model loaded successfully.")

        return self._model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"timesteps={self._total_timesteps}, "
            f"device={self._device!r})"
        )
        
    def save(
        self,
        path: str | None = None,
    ) -> None:
        """
        Saves the current PPO model.

        Parameters
        ----------
        path
            Optional destination path. If omitted, the experiment's
            default model path is used.

        Raises
        ------
        RuntimeError
            If the model has not been initialized.
        """

        if self._model is None:
            raise RuntimeError(
                "Cannot save an uninitialized model."
            )

        destination = (
            self._experiment.model_path
            if path is None
            else path
        )

        logger.info(
            "Saving PPO model to {}",
            destination,
        )

        self._model.save(str(destination))

        logger.success(
            "PPO model saved successfully."
        )

    def close(self) -> None:
        """
        Releases all environment resources.
        """

        logger.info(
            "Closing training environments."
        )

        self._train_env.close()
        self._eval_env.close()

    def __enter__(self) -> "PPOTrainer":
        """
        Enables usage with a context manager.
        """
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        Ensures environments are always closed.
        """
        self.close()