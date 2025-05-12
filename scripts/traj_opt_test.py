from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig

from vid2skill.common.trajopt.traj_opt import TrajectoryOptimizer
from vid2skill.common.trajopt.utils import load_dataset
from kinematic_retargeting_test import setup_environment, visualize_traj_with_meshcat


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    # Load kinematic feasible trajectory
    timestamp = "20250512_134425"
    file_name = "20200709_143211_839512060362.npy"
    kinematic_feasible_traj_path = (
        Path(__file__).parent.parent
        / f"data/traj_opt/kinematic_feasible_trajs/{timestamp}/{file_name}"
    )
    x_traj = np.load(kinematic_feasible_traj_path)

    # Create environment and visualize loaded kinematically feasible trajectory
    env, _ = setup_environment(cfg)
    # visualize_traj_with_meshcat(env.drake_system, x_traj)

    traj_optimizer = TrajectoryOptimizer(env, x_traj)
    dynamically_feasbible_x_traj = traj_optimizer.optimize()


if __name__ == "__main__":
    main()
