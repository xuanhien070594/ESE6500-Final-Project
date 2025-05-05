import numpy as np
from pydrake.solvers import MathematicalProgram
from franka_allegro_env_test import load_dataset
import hydra
import ipdb
import time
from omegaconf import DictConfig
from vid2skill.common.helper_functions.drake_helper_functions import make_env
from pydrake.autodiffutils import AutoDiffXd, ExtractValue
from pydrake.geometry import GeometryId
from pydrake.math import RotationMatrix
from pydrake.multibody.plant import MultibodyPlant_, MultibodyPlant
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.multibody.tree import RigidBody
from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    QuadraticConstraint,
)


@hydra.main(
    version_base=None,
    config_name="franka_allegro_drake_env",
    config_path="../configs/envs",
)
def main(cfg: DictConfig):
    dataset_name = "20200709_143211"
    camera_id = "839512060362"
    ref_obj_poses, ref_hand_poses = load_dataset(name=dataset_name, camera_id=camera_id)
    env = make_env(cfg)
    env.reset()
    env.render()

    n_q = env.drake_system.plant.num_positions()

    # Create a MathematicalProgram object and add decision variables
    prog = MathematicalProgram()
    prog.NewContinuousVariables(n_q, "q")

    for i in range(ref_hand_poses.shape[0]):
        env.drake_system.visualize_ref_hand_pose(ref_hand_poses[i])
        env.drake_system.visualize_ref_obj_pose(ref_obj_poses[i])
        time.sleep(0.1)


class KinematicRetargeting:
    def __init__(self, drake_system):
        self.drake_system = drake_system
        self.n_q = self.drake_system.plant.num_positions()
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(self.n_q, "q")

    def forward_kinematics_evaluator(self, q):
        if isinstance(q, AutoDiffXd):
            plant = self.drake_system._plant_ad
            context = self.drake_system._plant_ad_context
        else:
            plant = self.drake_system._plant
            context = self.drake_system._plant_context

        plant.SetPositions(context, q)
        plant.CalcForwardKinematics(context)


if __name__ == "__main__":
    main()
