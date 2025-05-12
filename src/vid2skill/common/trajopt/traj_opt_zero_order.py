from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import cmaes
import numpy as np
from loguru import logger
from scipy.interpolate import interp1d
from tqdm import tqdm


@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method."""

    n_samples: int = 50  # Number of samples per iteration
    n_elite: int = 5  # Number of elite samples to keep
    n_iterations: int = 50  # Maximum number of iterations
    smoothing_factor: float = 0.8  # Smoothing factor for mean and covariance updates
    initial_std: float = 0.05  # Initial standard deviation for sampling
    convergence_threshold: float = 1e-4  # Threshold for convergence


@dataclass
class CMAESConfig:
    """Configuration for The covariance matrix adaptation evolution strategy (CMA-ES)"""

    n_iterations: int = 50
    population_size: int = 60
    sigma: float = 0.1
    lower_bound: float = -1
    upper_bound: float = 1


@dataclass
class TrajOptCostWeights:
    w_franka_joint_pos_tracking: float = 0.01
    w_allegro_joint_pos_tracking: float = 0.01
    w_object_pose_tracking: float = 10
    w_franka_input_penalty: float = 0
    w_allegro_input_penalty: float = 0


class TrajectoryOptimizerBase(ABC):
    """Base class for trajectory optimization."""

    def __init__(
        self,
        env,
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
        self.env = env
        # self.n_steps = ref_trajectory.shape[0]
        self.subsample_interval = 10
        self.ref_trajectory = ref_trajectory
        self.n_steps = self.ref_trajectory.shape[0] // self.subsample_interval + 1
        self.full_n_steps = (self.n_steps - 1) * self.subsample_interval + 1
        self.cost_weights = cost_weights or TrajOptCostWeights()

        # Get state and control dimensions
        self.state_dim = env.drake_system.plant.num_multibody_states()
        self.control_dim = env.drake_system.plant.num_actuated_dofs()

        # Input scales
        self.control_scales = np.concatenate([[10, 10, 10, 10, 5, 5, 5], [0.5] * 16])

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

        for i in range(self.full_n_steps):
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

    def rollout(self, controls: np.ndarray) -> np.ndarray:
        """
        Simulate the system forward using the given controls.

        Args:
            controls: Control sequence (shape: n_steps x control_dim)

        Returns:
            np.ndarray: Resulting state trajectory
        """
        rescaled_controls = controls * self.control_scales
        interpolated_controls = self.interpolate_controls(rescaled_controls)
        trajectory = np.zeros((self.full_n_steps, self.state_dim))
        next_obs, _ = self.env.reset(specified_initial_state=self.ref_trajectory[0])

        for t in range(self.full_n_steps):
            trajectory[t] = next_obs
            next_obs, _, _, _, _ = self.env.step(interpolated_controls[t])
            self.env.render()

        return trajectory, interpolated_controls

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
        env,
        ref_trajectory: np.ndarray,
        config: Optional[CEMConfig] = None,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        super().__init__(env, ref_trajectory, cost_weights)
        self.config = config or CEMConfig()

        # Initialize mean and covariance for CEM
        self.mean = np.zeros((self.n_steps, self.control_dim))
        self.cov = np.eye(self.control_dim) * self.config.initial_std

    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Optimize the trajectory using Cross-Entropy Method.

        Returns:
            Tuple[np.ndarray, float]: Best trajectory and its cost
        """
        best_cost = float("inf")
        best_trajectory = None

        for iteration in range(self.config.n_iterations):
            logger.info(f"Iteration {iteration} of {self.config.n_iterations}")
            samples = np.random.multivariate_normal(
                self.mean.reshape(-1),
                np.kron(np.eye(self.n_steps), self.cov),
                size=self.config.n_samples,
            ).reshape(self.config.n_samples, self.n_steps, self.control_dim)
            samples = np.clip(samples, -1, 1)

            # Evaluate samples
            costs = []
            trajectories = []

            for i, sample in enumerate(samples):
                trajectory = self.rollout(sample)
                cost = self.compute_cost(trajectory, sample)
                costs.append(cost)
                trajectories.append(trajectory)

                if cost < best_cost:
                    best_cost = cost
                    best_trajectory = trajectory
                logger.info(f"Evaluating sample {i}, cost: {cost}")

            # Sort samples by cost
            sorted_indices = np.argsort(costs)
            elite_samples = samples[sorted_indices[: self.config.n_elite]]

            # Update mean and covariance
            new_mean = np.mean(elite_samples, axis=0)
            new_cov = np.cov(elite_samples.reshape(-1, self.control_dim).T)

            # Smooth updates
            self.mean = (
                1 - self.config.smoothing_factor
            ) * self.mean + self.config.smoothing_factor * new_mean
            self.cov = (
                1 - self.config.smoothing_factor
            ) * self.cov + self.config.smoothing_factor * new_cov

            # Check convergence
            if (
                iteration > 0
                and abs(best_cost - prev_best_cost) < self.config.convergence_threshold
            ):
                break

            prev_best_cost = best_cost

        return best_trajectory, best_cost


class TrajectoryOptimizerCMAES(TrajectoryOptimizerBase):
    """Trajectory optimization using CMA-ES."""

    def __init__(
        self,
        env,
        ref_trajectory: np.ndarray,
        config: Optional[CMAESConfig] = None,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        super().__init__(env, ref_trajectory, cost_weights)
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
            solutions = []
            for _ in tqdm(
                range(self.cmaes_solver.population_size),
                desc="Evaluating population",
                leave=False,
            ):
                flatten_u_traj = self.cmaes_solver.ask()
                u_traj = flatten_u_traj.reshape(self.n_steps, self.control_dim)
                rollout, interpolated_u_traj = self.rollout(u_traj)
                loss = self.compute_cost(rollout, interpolated_u_traj)
                solutions.append((flatten_u_traj, loss))
            self.cmaes_solver.tell(solutions)

            # Evaluate and log the cost of the mean solution
            mean_u_traj = self.cmaes_solver.mean.reshape(self.n_steps, self.control_dim)
            mean_rollout, mean_interpolated_u_traj = self.rollout(mean_u_traj)
            mean_cost = self.compute_cost(mean_rollout, mean_interpolated_u_traj)
            logger.info(f"Iteration {iteration}: Mean solution cost = {mean_cost:.4f}")

            if self.cmaes_solver.should_stop():
                break
        return self.cmaes_solver.mean
