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
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class SolverConfig:
    """Configuration for the optimization solver."""
    print_file: str = "snopt_log.txt"
    feasibility_tolerance: float = 1e-6
    optimality_tolerance: float = 1e-6
    
    def get_solver_options(self) -> Dict[str, any]:
        """Get solver options in the format expected by SNOPT."""
        return {
            "Print file": self.print_file,
            "Major feasibility tolerance": self.feasibility_tolerance,
            "Major optimality tolerance": self.optimality_tolerance,
        }

@dataclass
class TrackingWeights:
    """Configuration for tracking cost weights."""
    weights: List[float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = [
                10, 1, 1, 1,  # Thumb
                10, 1, 1, 1,  # Index
                3, 1, 1, 1,   # Middle
                3, 1, 1, 1,   # Ring
                3, 1, 1, 1,   # Little
            ]

class HandKeypointScaler:
    """Handles scaling of human hand keypoints to match robot hand dimensions."""
    
    def __init__(self, drake_system):
        self.drake_system = drake_system
        
    def compute_link_lengths(self, hand_keypoints_pos: np.ndarray) -> np.ndarray:
        """Compute the lengths between consecutive keypoints."""
        wrist_pos = hand_keypoints_pos[0]
        link_lengths = []
        
        n_joints_per_finger = 4
        n_fingers = (len(hand_keypoints_pos) - 1) // n_joints_per_finger
        
        for i in range(n_fingers):
            for j in range(1 + i * n_joints_per_finger, 1 + (i + 1) * n_joints_per_finger):
                if j == 1 + i * n_joints_per_finger:
                    link_lengths.append(np.linalg.norm(hand_keypoints_pos[j] - wrist_pos))
                else:
                    link_lengths.append(np.linalg.norm(hand_keypoints_pos[j] - hand_keypoints_pos[j - 1]))
                    
        return np.array(link_lengths)
    
    def scale_keypoints(self, ref_hand_keypoints_pos: np.ndarray, optimization: 'OptimizationProblem') -> np.ndarray:
        """Scale human hand keypoints to match robot hand dimensions."""
        cur_q = self.drake_system.plant.GetPositions(self.drake_system.plant_context)
        allegro_hand_keypoints_pos = optimization.forward_kinematics_evaluator(cur_q)
        
        human_link_lengths = self.compute_link_lengths(ref_hand_keypoints_pos)
        robot_link_lengths = self.compute_link_lengths(allegro_hand_keypoints_pos)
        
        ratio = np.mean([
            robot_link_length / human_link_length
            for human_link_length, robot_link_length in zip(human_link_lengths, robot_link_lengths)
        ])
        
        scaled_hand_keypoints_pos = ref_hand_keypoints_pos.copy()
        wrist_pos = ref_hand_keypoints_pos[0]
        
        n_joints_per_finger = 4
        n_fingers = (len(ref_hand_keypoints_pos) - 1) // n_joints_per_finger
        
        for i in range(n_fingers):
            for j in range(1 + i * n_joints_per_finger, 1 + (i + 1) * n_joints_per_finger):
                if j == 1 + i * n_joints_per_finger:
                    scaled_hand_keypoints_pos[j] = (ref_hand_keypoints_pos[j] - wrist_pos) + wrist_pos
                else:
                    scaled_hand_keypoints_pos[j] = (
                        ref_hand_keypoints_pos[j] - ref_hand_keypoints_pos[j - 1]
                    ) * ratio + scaled_hand_keypoints_pos[j - 1]
                    
        return scaled_hand_keypoints_pos

class OptimizationProblem:
    """Handles the setup and solving of the optimization problem."""
    
    def __init__(self, drake_system, n_q: int, solver_config: SolverConfig):
        self.drake_system = drake_system
        self.n_q = n_q
        self.solver_config = solver_config
        self.prog = None
        self.q = None
        self.solver = SnoptSolver()
        
    def initialize(self):
        """Initialize the optimization problem."""
        self.prog = MathematicalProgram()
        self.q = self.prog.NewContinuousVariables(self.n_q, "q")
        
        for option, value in self.solver_config.get_solver_options().items():
            self.prog.SetSolverOption(self.solver.solver_id(), option, value)
            
    def add_joint_limit_constraints(self):
        """Add joint limit constraints."""
        q_min = self.drake_system.plant.GetPositionLowerLimits()
        q_max = self.drake_system.plant.GetPositionUpperLimits()
        self.prog.AddBoundingBoxConstraint(q_min, q_max, self.q)
        
    def add_object_pose_constraint(self, ref_obj_pose: np.ndarray):
        """Add object pose equality constraint."""
        self.prog.AddBoundingBoxConstraint(ref_obj_pose, ref_obj_pose, self.q[-7:])
        
    def add_tracking_cost(self, tracking_weights: TrackingWeights, scaled_ref_hand_keypoints_pos: np.ndarray):
        """Add tracking cost to the optimization problem."""
        def tracking_cost_evaluator(vars):
            cur_q = vars[:self.n_q]
            cur_hand_keypoints_pos = self.forward_kinematics_evaluator(cur_q)
            
            cost = 0
            for i in range(len(cur_hand_keypoints_pos)):
                tracking_error = cur_hand_keypoints_pos[i] - scaled_ref_hand_keypoints_pos[i]
                cost += tracking_weights.weights[i] * tracking_error.T @ tracking_error
            return cost
            
        self.prog.AddCost(tracking_cost_evaluator, vars=self.q)
        
    def forward_kinematics_evaluator(self, q: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint positions."""
        if isinstance(q[0], AutoDiffXd):
            plant = self.drake_system.plant_ad
            context = self.drake_system.plant_ad_context
        else:
            plant = self.drake_system.plant
            context = self.drake_system.plant_context
            
        plant.SetPositions(context, q)
        hand_keypoints_pos = []
        
        for name in self.drake_system.hand_body_names:
            hand_keypoints_pos.append(
                plant.EvalBodyPoseInWorld(context, plant.GetBodyByName(name)).translation()
            )
            
        return np.array(hand_keypoints_pos)
        
    def solve(self, prev_sol: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve the optimization problem."""
        result = self.solver.Solve(self.prog, prev_sol)
        if not result.is_success():
            ipdb.set_trace()
            raise RuntimeError("Solver failed to find a solution.")
        return result.GetSolution(self.q)

class KinematicRetargeting:
    """Main class for kinematic retargeting from human hand to robot hand."""
    
    def __init__(
        self, 
        drake_system, 
        ref_hand_keypoints_pos: np.ndarray, 
        ref_obj_pose: np.ndarray,
        solver_config: Optional[SolverConfig] = None,
        tracking_weights: Optional[TrackingWeights] = None
    ):
        self.drake_system = drake_system
        self.n_q = self.drake_system.plant.num_positions()
        
        # Convert object pose to quaternion format
        self.ref_obj_pose = np.concatenate([
            RotationMatrix(ref_obj_pose[:3, :3]).ToQuaternion().wxyz(),
            ref_obj_pose[:3, 3],
        ])
        
        # Initialize components
        self.solver_config = solver_config or SolverConfig()
        self.tracking_weights = tracking_weights or TrackingWeights()
        self.optimization = OptimizationProblem(drake_system, self.n_q, self.solver_config)
        self.keypoint_scaler = HandKeypointScaler(drake_system)
        
        # Setup optimization problem
        self.optimization.initialize()
        
        # Scale reference hand keypoints
        self.scaled_ref_hand_keypoints_pos = self.keypoint_scaler.scale_keypoints(
            ref_hand_keypoints_pos, 
            self.optimization
        )
        
        # Add constraints and costs
        self.optimization.add_tracking_cost(self.tracking_weights, self.scaled_ref_hand_keypoints_pos)
        self.optimization.add_object_pose_constraint(self.ref_obj_pose)
        self.optimization.add_joint_limit_constraints()
        
    def solve(self, prev_sol: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve the kinematic retargeting optimization problem."""
        return self.optimization.solve(prev_sol)
