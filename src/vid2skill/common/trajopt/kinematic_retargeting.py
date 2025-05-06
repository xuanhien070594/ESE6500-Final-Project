import numpy as np
from pydrake.solvers import MathematicalProgram
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
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
from pydrake.solvers import (
    MathematicalProgram,
    MathematicalProgramResult,
    SnoptSolver,
    QuadraticConstraint,
)


class KinematicRetargeting:
    def __init__(self, drake_system, ref_hand_pose, ref_obj_pose):
        self.drake_system = drake_system
        self.n_q = self.drake_system.plant.num_positions()
        self.ref_obj_pose = np.concatenate(
            [
                RotationMatrix(ref_obj_pose[:3, :3]).ToQuaternion().wxyz(),
                ref_obj_pose[:3, 3],
            ]
        )
        self.ref_hand_pose = ref_hand_pose

        self.contact_pairs, _ = self.drake_system.get_contact_pairs()

        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(self.n_q, "q")

        self.add_tracking_cost()
        self.add_equality_constraint_for_obj_pose()
        self.add_no_penetration_constraint()
        self.add_joint_limits()

        self.solver = SnoptSolver()
        self.solver_options = {
            "Print file": "snopt_log.txt",
            "Major feasibility tolerance": 1e-6,
            "Major optimality tolerance": 1e-6,
        }
        for option, value in self.solver_options.items():
            self.prog.SetSolverOption(self.solver.solver_id(), option, value)

        self.scaled_ref_hand_pose = self.scaled_human_hand_pose()

        self.weights = [3, 1, 1, 1, 10, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3]

    def scaled_human_hand_pose(self):
        cur_q = self.drake_system.plant.GetPositions(self.drake_system.plant_context)
        allegro_hand_pose = self.forward_kinematics_evaluator(cur_q)
        human_link_lengths = self.compute_link_lengths(self.ref_hand_pose)
        robot_link_lengths = self.compute_link_lengths(allegro_hand_pose)
        ratio = np.mean(
            [
                robot_link_length / human_link_length
                for human_link_length, robot_link_length in zip(
                    human_link_lengths, robot_link_lengths
                )
            ]
        )
        scaled_hand_pose = self.ref_hand_pose.copy()
        wrist_pose = self.ref_hand_pose[0]

        n_joints_per_finger = 4
        n_fingers = (len(self.ref_hand_pose) - 1) // n_joints_per_finger
        for i in range(n_fingers):
            for j in range(
                1 + i * n_joints_per_finger, 1 + (i + 1) * n_joints_per_finger
            ):
                if j == 1 + i * n_joints_per_finger:
                    scaled_hand_pose[j] = (
                        self.ref_hand_pose[j] - wrist_pose
                    ) * 1.2 + wrist_pose
                else:
                    scaled_hand_pose[j] = (
                        self.ref_hand_pose[j] - self.ref_hand_pose[j - 1]
                    ) * ratio + scaled_hand_pose[j - 1]
        return scaled_hand_pose

    def compute_link_lengths(self, hand_pose):
        wrist_pose = hand_pose[0]
        link_lengths = []

        n_joints_per_finger = 4
        n_fingers = (len(hand_pose) - 1) // n_joints_per_finger
        for i in range(n_fingers):
            for j in range(
                1 + i * n_joints_per_finger, 1 + (i + 1) * n_joints_per_finger
            ):
                if j == 1 + i * n_joints_per_finger:
                    link_lengths.append(np.linalg.norm(hand_pose[j] - wrist_pose))
                else:
                    link_lengths.append(np.linalg.norm(hand_pose[j] - hand_pose[j - 1]))
        link_lengths = np.array(link_lengths)
        return link_lengths

    def forward_kinematics_evaluator(self, q):
        if isinstance(q[0], AutoDiffXd):
            plant = self.drake_system.plant_ad
            context = self.drake_system.plant_ad_context
        else:
            plant = self.drake_system.plant
            context = self.drake_system.plant_context

        plant.SetPositions(context, q)
        hand_pose = []

        for name in self.drake_system.hand_body_names:
            hand_pose.append(
                plant.EvalBodyPoseInWorld(
                    context, plant.GetBodyByName(name)
                ).translation()
            )
        hand_pose = np.array(hand_pose)
        return hand_pose

    def tracking_cost_evaluator(self, vars):
        cur_q = vars[: self.n_q]
        cur_hand_pose = self.forward_kinematics_evaluator(cur_q)

        cost = 0
        for i in range(len(cur_hand_pose)):
            tracking_error = cur_hand_pose[i] - self.scaled_ref_hand_pose[i]
            cost += self.weights[i] * tracking_error.T @ tracking_error
        return cost

    def add_tracking_cost(self):
        self.prog.AddCost(
            self.tracking_cost_evaluator,
            vars=self.q,
        )

    # def signed_distance_evaluator(self, q):
    #     if isinstance(q[0], AutoDiffXd):
    #         plant = self.drake_system.plant_ad
    #         context = self.drake_system.plant_ad_context
    #     else:
    #         plant = self.drake_system.plant
    #         context = self.drake_system.plant_context

    #     plant.SetPositions(context, self.drake_system.franka_allegro_model_instance, q)
    #     plant.SetPositions(
    #         context, self.drake_system.mustard_bottle_model_instance, self.ref_obj_pose
    #     )

    #     query_object = plant.get_geometry_query_input_port().Eval(context)

    #     sdf = []
    #     for geom_pair in self.contact_pairs:
    #         sdf.append(
    #             query_object.ComputeSignedDistancePairClosestPoints(*geom_pair).distance
    #         )
    #     sdf = np.array(sdf)
    #     return sdf

    # def signed_distance_evaluator_helper(self, vars):
    #     cur_q = vars[: self.n_q]
    #     return self.signed_distance_evaluator(cur_q)

    def add_no_penetration_constraint(self):
        min_dist_constraint = MinimumDistanceLowerBoundConstraint(
            self.drake_system.plant, -1e-3, self.drake_system.plant_context
        )
        self.prog.AddConstraint(
            min_dist_constraint,
            vars=self.q,
        )

    def add_joint_limits(self):
        q_min = self.drake_system.plant.GetPositionLowerLimits()
        q_max = self.drake_system.plant.GetPositionUpperLimits()
        self.prog.AddBoundingBoxConstraint(q_min, q_max, self.q)

    def add_equality_constraint_for_obj_pose(self):
        self.prog.AddBoundingBoxConstraint(
            self.ref_obj_pose, self.ref_obj_pose, self.q[-7:]
        )

    def solve(self, prev_sol=None):
        result = self.solver.Solve(self.prog, prev_sol)
        if not result.is_success():
            ipdb.set_trace()
            raise RuntimeError("Solver failed to find a solution.")
        return result.GetSolution(self.q)
