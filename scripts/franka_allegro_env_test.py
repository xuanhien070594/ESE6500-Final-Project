import numpy as np
from pathlib import Path
import hydra
import ipdb
import yaml
import time
from omegaconf import DictConfig
from vid2skill.common.helper_functions.drake_helper_functions import make_env
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
    # ref_hand_poses_exclude_pinky = ref_hand_poses[:, :-4, :]
    env = make_env(cfg)
    cur_state, _ = env.reset()
    env.render()

    env.drake_system.plant.SetPositionsAndVelocities(
        env.drake_system.plant_context, cur_state
    )

    prev_sol = None
    for i in range(ref_hand_poses.shape[0]):
        env.drake_system.visualize_hand_pose(ref_hand_poses[i])
        env.drake_system.visualize_ref_obj_pose(ref_obj_poses[i])

        # perform kinematic retargeting for single timestep
        kinematic_retargeting = KinematicRetargeting(
            env.drake_system, ref_hand_poses_exclude_pinky[i], ref_obj_poses[i]
        )
        # sdf = kinematic_retargeting.signed_distance_evaluator(cur_q)
        sol = kinematic_retargeting.solve(prev_sol)
        prev_sol = sol
        optimized_hand_pose = kinematic_retargeting.forward_kinematics_evaluator(sol)
        env.drake_system.visualize_hand_pose(optimized_hand_pose, is_human_hand=False)

        # visualize robot state
        env.drake_system.plant.SetPositions(
            env.drake_system.plant_context,
            sol,
        )
        # env.reset(
        #     specified_initial_state=env.drake_system.plant.GetPositionsAndVelocities(
        #         env.drake_system.plant_context
        #     )
        # )
        # ipdb.set_trace()
        time.sleep(0.3)


if __name__ == "__main__":
    main()
