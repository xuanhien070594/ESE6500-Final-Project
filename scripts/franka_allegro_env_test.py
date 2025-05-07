import hydra
import numpy as np
from omegaconf import DictConfig

from vid2skill.common.helper_functions.drake_helper_functions import (
    make_env,
    visualize_traj_with_meshcat,
)
from vid2skill.common.trajopt.kinematic_retargeting import KinematicRetargeting
from vid2skill.common.trajopt.utils import load_dataset


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    dataset_name = "20200709_143211"
    camera_id = "839512060362"
    ref_obj_poses, ref_hand_poses = load_dataset(name=dataset_name, camera_id=camera_id)
    ref_hand_poses_exclude_pinky = np.concatenate(
        [ref_hand_poses[:, :-8, :], ref_hand_poses[:, -4:, :]], axis=1
    )

    # Load the environment and set the home position
    env = make_env(cfg)
    cur_state, _ = env.reset()
    env.render()

    # Warm-start solution for accelerating SQP solver
    prev_sol = env.drake_system.plant.GetPositions(env.drake_system.plant_context)

    # container to store trajectory
    x_traj = []

    for t in range(ref_hand_poses.shape[0]):
        env.drake_system.visualize_hand_pose(ref_hand_poses[t])
        env.drake_system.visualize_ref_obj_pose(ref_obj_poses[t])

        # perform kinematic retargeting for single timestep t
        kinematic_retargeting = KinematicRetargeting(
            env.drake_system, ref_hand_poses_exclude_pinky[t], ref_obj_poses[t]
        )
        sol = kinematic_retargeting.solve(prev_sol)
        prev_sol = sol
        optimized_hand_pose = kinematic_retargeting.forward_kinematics_evaluator(sol)
        env.drake_system.visualize_hand_pose(optimized_hand_pose, is_human_hand=False)

        cur_state = np.concatenate(
            [sol, env.drake_system.plant.GetVelocities(env.drake_system.plant_context)]
        )
        x_traj.append(cur_state)

    x_traj = np.array(x_traj)

    visualize_traj_with_meshcat(env.drake_system, x_traj)


if __name__ == "__main__":
    main()
