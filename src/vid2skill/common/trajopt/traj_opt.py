import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class CEMConfig:
    """Configuration for Cross-Entropy Method."""

    n_samples: int = 50  # Number of samples per iteration
    n_elite: int = 5  # Number of elite samples to keep
    n_iterations: int = 50  # Maximum number of iterations
    smoothing_factor: float = 0.1  # Smoothing factor for mean and covariance updates
    initial_std: float = 0.05  # Initial standard deviation for sampling
    convergence_threshold: float = 1e-4  # Threshold for convergence


@dataclass
class TrajOptCostWeights:
    w_franka_joint_pos_tracking: float = 0.01
    w_allegro_joint_pos_tracking: float = 0.01
    w_object_pose_tracking: float = 10
    w_franka_input_penalty: float = 0.1
    w_allegro_input_penalty: float = 0.01


class TrajectoryOptimizer:
    """Trajectory optimization using Cross-Entropy Method."""

    def __init__(
        self,
        env,
        ref_trajectory: np.ndarray,
        config: Optional[CEMConfig] = None,
        cost_weights: Optional[TrajOptCostWeights] = None,
    ):
        """
        Initialize the trajectory optimizer.

        Args:
            drake_system: Drake system for simulation
            ref_trajectory: Reference trajectory to track (shape: n_steps x state_dim)
            n_steps: Number of timesteps in the trajectory
            config: Configuration for CEM
            cost_weights: Cost weights for the trajectory optimization
        """
        self.env = env
        self.ref_trajectory = ref_trajectory[:6]
        # self.n_steps = ref_trajectory.shape[0]
        self.n_steps = 6
        self.config = config or CEMConfig()
        self.cost_weights = cost_weights or TrajOptCostWeights()

        # Get state and control dimensions
        self.state_dim = env.drake_system.plant.num_multibody_states()
        self.control_dim = env.drake_system.plant.num_actuated_dofs()

        # Initialize mean and covariance for CEM
        self.mean = np.zeros((self.n_steps, self.control_dim))
        self.cov = np.eye(self.control_dim) * self.config.initial_std

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

        for i in range(self.n_steps):
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

    def rollout(self, controls: np.ndarray) -> np.ndarray:
        """
        Simulate the system forward using the given controls.

        Args:
            controls: Control sequence (shape: n_steps x control_dim)

        Returns:
            np.ndarray: Resulting state trajectory
        """
        trajectory = np.zeros((self.n_steps, self.state_dim))
        next_obs, _ = self.env.reset(specified_initial_state=self.ref_trajectory[0])

        for t in range(self.n_steps - 1):
            trajectory[t] = next_obs
            next_obs, _, _, _, _ = self.env.step(controls[t])
            self.env.render()

        return trajectory

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

            # # Check convergence
            # if (
            #     iteration > 0
            #     and abs(best_cost - prev_best_cost) < self.config.convergence_threshold
            # ):
            #     break

            prev_best_cost = best_cost

        return best_trajectory, best_cost
