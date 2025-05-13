import gc
import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cmaes
import numpy as np
from loguru import logger
from scipy.interpolate import interp1d
from tqdm import tqdm

from vid2skill.common.helper_functions.drake_helper_functions import (
    visualize_traj_with_meshcat,
)


@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method."""

    n_samples: int = 50  # Number of samples per iteration
    n_elite: int = 5  # Number of elite samples to keep
    n_iterations: int = 200  # Maximum number of iterations
    smoothing_factor: float = 0.5  # Smoothing factor for mean and covariance updates
    initial_std: float = 0.3  # Initial standard deviation for sampling
    convergence_threshold: float = 1e-4  # Threshold for convergence
    n_workers: int = 10  # Number of workers for parallel execution


@dataclass
class CMAESConfig:
    """Configuration for The covariance matrix adaptation evolution strategy (CMA-ES)"""

    n_iterations: int = 50
    population_size: int = 60
    sigma: float = 0.3
    lower_bound: float = -1
    upper_bound: float = 1
    n_workers: int = 10  # Number of workers for parallel execution


@dataclass
class TrajOptCostWeights:
    w_franka_joint_pos_tracking: float = 0.01
    w_allegro_joint_pos_tracking: float = 0.01
    w_object_pose_tracking: float = 10
    w_franka_input_penalty: float = 0
    w_allegro_input_penalty: float = 0
    w_bounds_violation_penalty: float = 1000.0


class TrajectoryOptimizerBase(ABC):
    """Base class for trajectory optimization."""

    def __init__(
        self,
        create_env_w_visualizer_fn: Callable,
        create_env_wo_visualizer_fn: Callable,
        ref_trajectory: np.ndarray,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        """
        Initialize the trajectory optimizer.

        Args:
            env: Environment for simulation
            ref_trajectory: Reference trajectory to track (shape: n_steps x state_dim)
            cost_weights: Cost weights for the trajectory optimization
        """
        self.create_env_w_visualizer_fn = create_env_w_visualizer_fn
        self.create_env_wo_visualizer_fn = create_env_wo_visualizer_fn
        self.subsample_interval = 2
        self.ref_trajectory = ref_trajectory
        self.n_steps = (self.ref_trajectory.shape[0] - 1) // self.subsample_interval + 1
        self.full_n_steps = (self.n_steps - 1) * self.subsample_interval + 1
        self.cost_weights = cost_weights or TrajOptCostWeights()

        self.control_dim = 23

        # Input scales
        self.control_scales = np.concatenate([[5, 5, 5, 5, 2, 2, 2], [0.2] * 16])

    def compute_cost(self, trajectory: np.ndarray, control_inputs: np.ndarray) -> float:
        """
        Compute the L2 cost between the trajectory and reference trajectory.

        Args:
            trajectory: Generated trajectory (shape: n_steps x state_dim)
            control_inputs: Control inputs that were used to generate the trajectory (shape: n_steps x control_dim)

        Returns:
            float: Total cost
        """
        cost = 0

        # Define state bounds parameters
        state_margin = 0.2  # margin in radians

        for i in range(self.full_n_steps):
            # Compute bounds for current timestep
            state_lower_bounds = self.ref_trajectory[i, :23] - state_margin
            state_upper_bounds = self.ref_trajectory[i, :23] + state_margin

            # Add bounds violation penalty
            state_violations = np.maximum(
                0, state_lower_bounds - trajectory[i, :23]
            ) + np.maximum(0, trajectory[i, :23] - state_upper_bounds)
            bounds_cost = np.sum(state_violations)

            franka_joint_pos_tracking_cost = (
                np.linalg.norm(trajectory[i, :7] - self.ref_trajectory[i, :7]) ** 2
            )
            allegro_joint_pos_tracking_cost = (
                np.linalg.norm(trajectory[i, 7:23] - self.ref_trajectory[i, 7:23]) ** 2
            )
            object_pos_tracking_cost = (
                np.linalg.norm(trajectory[i, 23:30] - self.ref_trajectory[i, 23:30])
                ** 2
            )
            single_step_cost = (
                self.cost_weights.w_franka_joint_pos_tracking
                * franka_joint_pos_tracking_cost
                + self.cost_weights.w_allegro_joint_pos_tracking
                * allegro_joint_pos_tracking_cost
                + self.cost_weights.w_object_pose_tracking * object_pos_tracking_cost
                + self.cost_weights.w_bounds_violation_penalty * bounds_cost
            )

            if i == self.n_steps - 1:
                single_step_cost *= 10
            else:
                single_step_cost += (
                    self.cost_weights.w_franka_input_penalty
                    * np.linalg.norm(control_inputs[i, :7]) ** 2
                    + self.cost_weights.w_allegro_input_penalty
                    * np.linalg.norm(control_inputs[i, 7:23]) ** 2
                )

            cost += single_step_cost

        return cost

    def interpolate_controls(self, subsampled_controls: np.ndarray) -> np.ndarray:
        """
        Interpolate subsampled controls to full trajectory length.

        Args:
            subsampled_controls: Control sequence for subsampled points (shape: n_subsample x control_dim)

        Returns:
            np.ndarray: Interpolated control sequence (shape: n_steps x control_dim)
        """
        original_timesteps = np.linspace(0, 1, subsampled_controls.shape[0])
        full_timesteps = np.linspace(0, 1, self.full_n_steps)
        interpolated_controls = np.zeros((self.full_n_steps, self.control_dim))
        for i in range(self.control_dim):
            f = interp1d(original_timesteps, subsampled_controls[:, i], kind="linear")
            interpolated_controls[:, i] = f(full_timesteps)
        return interpolated_controls

    def rollout(self, env, controls: np.ndarray) -> np.ndarray:
        """
        Simulate the system forward using the given controls.

        Args:
            env: Environment for simulation
            controls: Control sequence (shape: n_steps x control_dim)

        Returns:
            np.ndarray: Resulting state trajectory
        """
        state_dim = env.drake_system.plant.num_multibody_states()

        # Since CMA-ES returns normalized controls between -1 and 1, we need to rescale them
        # to the actual control scales
        rescaled_controls = controls * self.control_scales

        # Interpolate the controls to the full trajectory length
        interpolated_controls = self.interpolate_controls(rescaled_controls)

        # Simulate the system forward
        trajectory = np.zeros((self.full_n_steps, state_dim))
        next_obs, _ = env.reset(specified_initial_state=self.ref_trajectory[0])

        for t in range(self.full_n_steps):
            trajectory[t] = next_obs
            next_obs, _, _, _, _ = env.step(interpolated_controls[t])

        return trajectory, interpolated_controls

    def worker_to_execute_and_eval_rollout(self, controls) -> float:
        """
        Execute the rollout and evaluate the cost.

        Args:
            controls: Flattened control sequence (shape: n_steps * control_dim)
        """
        env = self.create_env_wo_visualizer_fn()
        controls = controls.reshape(self.n_steps, self.control_dim)
        rollout, interpolated_controls = self.rollout(env, controls)
        cost = self.compute_cost(rollout, interpolated_controls)
        return cost

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Optimize the trajectory using the specific optimization method.

        Returns:
            Tuple[np.ndarray, float]: Best trajectory and its cost
        """
        pass


class TrajectoryOptimizerCEM(TrajectoryOptimizerBase):
    """Trajectory optimization using Cross-Entropy Method."""

    def __init__(
        self,
        create_env_w_visualizer_fn: Callable,
        create_env_wo_visualizer_fn: Callable,
        ref_trajectory: np.ndarray,
        config: Optional[CEMConfig] = None,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        super().__init__(
            create_env_w_visualizer_fn,
            create_env_wo_visualizer_fn,
            ref_trajectory,
            cost_weights,
        )
        self.config = config or CEMConfig()

        # Initialize mean and covariance for CEM
        self.mean = np.zeros(self.n_steps * self.control_dim)
        self.cov = np.eye(self.n_steps * self.control_dim) * self.config.initial_std

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Optimize the trajectory using Cross-Entropy Method.

        Returns:
            Tuple[np.ndarray, float]: Mean trajectory and its cost
        """
        prev_mean_cost = float("inf")

        for iteration in tqdm(range(self.config.n_iterations), desc="CEM Optimization"):
            logger.info(f"Iteration {iteration} of {self.config.n_iterations}")
            samples = np.random.multivariate_normal(
                self.mean.reshape(-1),
                self.cov,
                size=self.config.n_samples,
            )
            samples = np.clip(samples, -1, 1)

            # Evaluate samples in parallel
            with mp.Pool(processes=self.config.n_workers) as pool:
                costs = pool.map(
                    self.worker_to_execute_and_eval_rollout,
                    [sample.reshape(-1) for sample in samples],
                )

            # Sort samples by cost
            sorted_indices = np.argsort(costs)
            elite_samples = samples[sorted_indices[: self.config.n_elite]]

            # Update mean and covariance
            new_mean = np.mean(elite_samples, axis=0)
            new_cov = np.cov(elite_samples.T)

            # Smooth updates
            self.mean = (1 - self.config.smoothing_factor) * self.mean.reshape(
                -1
            ) + self.config.smoothing_factor * new_mean
            self.cov = (
                1 - self.config.smoothing_factor
            ) * self.cov + self.config.smoothing_factor * new_cov

            # Evaluate the mean trajectory
            if iteration % 50 == 0:
                eval_env = self.create_env_w_visualizer_fn()
            else:
                eval_env = self.create_env_wo_visualizer_fn()
            mean_trajectory, interpolated_controls = self.rollout(
                eval_env, self.mean.reshape(self.n_steps, self.control_dim)
            )
            mean_cost = self.compute_cost(mean_trajectory, interpolated_controls)
            logger.info(f"Iteration {iteration}: Mean solution cost = {mean_cost:.4f}")

            if eval_env.drake_system.meshcat is not None:
                visualize_traj_with_meshcat(eval_env.drake_system, mean_trajectory)
                input("Press Enter to continue...")
                # Prevent destructor of meshcat from being called in non-main thread
                del eval_env
                gc.collect()

            # # Check convergence
            # if (
            #     iteration > 0
            #     and abs(mean_cost - prev_mean_cost) < self.config.convergence_threshold
            # ):
            #     break

            prev_mean_cost = mean_cost

        return mean_trajectory, mean_cost


class TrajectoryOptimizerCMAES(TrajectoryOptimizerBase):
    """Trajectory optimization using CMA-ES."""

    def __init__(
        self,
        create_env_w_visualizer_fn: Callable,
        create_env_wo_visualizer_fn: Callable,
        ref_trajectory: np.ndarray,
        config: Optional[CMAESConfig] = None,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        super().__init__(
            create_env_w_visualizer_fn,
            create_env_wo_visualizer_fn,
            ref_trajectory,
            cost_weights,
        )
        self.config = config or CMAESConfig()
        self.cmaes_solver = cmaes.CMA(
            mean=np.zeros(self.n_steps * self.control_dim),
            sigma=self.config.sigma,
            bounds=np.vstack(
                [
                    np.full(self.n_steps * self.control_dim, self.config.lower_bound),
                    np.full(self.n_steps * self.control_dim, self.config.upper_bound),
                ]
            ).T,
            population_size=self.config.population_size,
        )

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Optimize the trajectory using CMA-ES.

        Returns:
            Tuple[np.ndarray, float]: Best trajectory and its cost
        """
        for iteration in tqdm(
            range(self.config.n_iterations), desc="CMA-ES Optimization"
        ):
            # Get all solutions for this iteration
            samples = [
                self.cmaes_solver.ask()
                for _ in range(self.cmaes_solver.population_size)
            ]

            # Evaluate solutions in parallel using pool.starmap
            with mp.Pool(processes=self.config.n_workers) as pool:
                results = pool.starmap(
                    self.worker_to_execute_and_eval_rollout,
                    [(sample,) for sample in samples],
                )

            # Extract costs from results
            solutions = [
                (sample, cost.item()) for sample, cost in zip(samples, results)
            ]

            # Tell results to CMA-ES solver
            self.cmaes_solver.tell(solutions)

            # Evaluate and log the cost of the mean solution
            try:
                eval_env = self.create_env_w_visualizer_fn()
                mean_u_traj = self.cmaes_solver.mean.reshape(
                    self.n_steps, self.control_dim
                )
                mean_rollout, mean_interpolated_u_traj = self.rollout(
                    eval_env, mean_u_traj
                )
                mean_cost = self.compute_cost(mean_rollout, mean_interpolated_u_traj)
                visualize_traj_with_meshcat(eval_env.drake_system, mean_rollout)
                logger.info(
                    f"Iteration {iteration}: Mean solution cost = {mean_cost:.4f}"
                )
                input("Press Enter to continue...")
            finally:
                del eval_env
                gc.collect()

            if self.cmaes_solver.should_stop():
                break
        return self.cmaes_solver.mean
