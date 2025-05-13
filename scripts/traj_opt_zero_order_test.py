from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import copy
import hydra
import numpy as np
from omegaconf import DictConfig

from vid2skill.common.trajopt.traj_opt_zero_order import (
    TrajectoryOptimizerCEM,
    TrajectoryOptimizerCMAES,
)
from vid2skill.common.trajopt.utils import set_random_seed
from kinematic_retargeting_test import setup_environment
from functools import partial


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

    # Set random seed
    set_random_seed(0)

    # Create functions to create environments with and without visualizer
    create_env_w_visualizer_fn = partial(setup_environment, cfg)
    cfg_wo_visualizer = copy.deepcopy(cfg)
    cfg_wo_visualizer.configs.enable_visualizer = False
    create_env_wo_visualizer_fn = partial(setup_environment, cfg_wo_visualizer)

    # Create trajectory optimizer
    traj_optimizer = TrajectoryOptimizerCEM(
        create_env_w_visualizer_fn, create_env_wo_visualizer_fn, x_traj
    )
    dynamically_feasbible_x_traj = traj_optimizer.optimize()


if __name__ == "__main__":
    main()
