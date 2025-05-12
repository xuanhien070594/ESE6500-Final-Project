from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from vid2skill.common.helper_functions.drake_helper_functions import (
    make_env,
    visualize_traj_with_meshcat,
)
from vid2skill.common.trajopt.kinematic_retargeting import KinematicRetargeting
from vid2skill.common.trajopt.utils import load_dataset


def load_and_preprocess_dataset(
    dataset_name: str, camera_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the dataset for hand poses and object poses.

    Args:
        dataset_name: Name of the dataset to load
        camera_id: ID of the camera used for recording

    Returns:
        Tuple containing:
        - Reference object trajectory
        - Original hand keypoints trajectory
        - Preprocessed hand keypoints trajectory (excluding pinky)
    """
    ref_obj_traj, ref_hand_keypoints_traj = load_dataset(
        name=dataset_name, camera_id=camera_id
    )
    # Exclude pinky finger from hand poses
    ref_hand_keypoints_traj_exclude_pinky = np.concatenate(
        [ref_hand_keypoints_traj[:, :-8, :], ref_hand_keypoints_traj[:, -4:, :]], axis=1
    )
    return ref_obj_traj, ref_hand_keypoints_traj, ref_hand_keypoints_traj_exclude_pinky


def setup_environment(cfg: DictConfig) -> Tuple[object, np.ndarray]:
    """
    Initialize the environment and get initial state.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple containing environment object and initial state
    """
    env = make_env(cfg)
    cur_state, _ = env.reset()
    env.render()
    return env, cur_state


def perform_kinematic_retargeting(
    env: object,
    ref_hand_keypoints: np.ndarray,
    ref_hand_keypoints_original: np.ndarray,
    ref_obj_pose: np.ndarray,
    prev_sol: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform kinematic retargeting for a single timestep.

    Args:
        env: Environment object
        ref_hand_keypoints: Reference hand keypoints for current timestep (excluding pinky)
        ref_hand_keypoints_original: Original reference hand keypoints for visualization
        ref_obj_pose: Reference object pose for current timestep
        prev_sol: Previous solution for warm-start

    Returns:
        Tuple containing current state and solution
    """
    kinematic_retargeting = KinematicRetargeting(
        env.drake_system, ref_hand_keypoints, ref_obj_pose
    )
    sol = kinematic_retargeting.solve(prev_sol)
    optimized_hand_keypoints = (
        kinematic_retargeting.optimization.forward_kinematics_evaluator(sol)
    )

    # Visualize poses using original keypoints for reference visualization
    env.drake_system.visualize_hand_pose(ref_hand_keypoints_original)
    env.drake_system.visualize_ref_obj_pose(ref_obj_pose)
    env.drake_system.visualize_hand_pose(optimized_hand_keypoints, is_human_hand=False)

    cur_state = np.concatenate(
        [sol, env.drake_system.plant.GetVelocities(env.drake_system.plant_context)]
    )
    return cur_state, sol


def generate_trajectory(
    env: object,
    ref_hand_keypoints_traj: np.ndarray,
    ref_hand_keypoints_traj_original: np.ndarray,
    ref_obj_traj: np.ndarray,
) -> np.ndarray:
    """
    Generate trajectory by performing kinematic retargeting for each timestep.

    Args:
        env: Environment object
        ref_hand_keypoints_traj: Reference hand keypoints trajectory for all timesteps (excluding pinky)
        ref_hand_keypoints_traj_original: Original reference hand keypoints trajectory for visualization
        ref_obj_traj: Reference object trajectory for all timesteps

    Returns:
        Array containing the generated trajectory
    """
    x_traj: List[np.ndarray] = []
    prev_sol = env.drake_system.plant.GetPositions(env.drake_system.plant_context)

    for t in range(ref_hand_keypoints_traj.shape[0]):
        cur_state, prev_sol = perform_kinematic_retargeting(
            env,
            ref_hand_keypoints_traj[t],
            ref_hand_keypoints_traj_original[t],
            ref_obj_traj[t],
            prev_sol,
        )
        x_traj.append(cur_state)

    return np.array(x_traj)


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    # Load and preprocess dataset
    dataset_name = "20200709_143211"
    camera_id = "839512060362"

    # Create directory for kinematic feasible trajectories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kinematic_feasible_traj_path = (
        Path(__file__).parent.parent
        / f"data/traj_opt/kinematic_feasible_trajs/{timestamp}"
        / f"{dataset_name}_{camera_id}.npy"
    )
    kinematic_feasible_traj_path.parent.mkdir(parents=True, exist_ok=True)
    ref_obj_traj, ref_hand_keypoints_traj_original, ref_hand_keypoints_traj = (
        load_and_preprocess_dataset(dataset_name, camera_id)
    )

    # Setup environment
    env, _ = setup_environment(cfg)

    # Generate trajectory
    x_traj = generate_trajectory(
        env, ref_hand_keypoints_traj, ref_hand_keypoints_traj_original, ref_obj_traj
    )
    np.save(kinematic_feasible_traj_path, x_traj)

    # Visualize final trajectory
    visualize_traj_with_meshcat(env.drake_system, x_traj)


if __name__ == "__main__":
    main()
