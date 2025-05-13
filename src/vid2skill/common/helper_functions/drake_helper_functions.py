import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
from pydrake.all import RevoluteJoint
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import (
    Box,
    Cylinder,
    Meshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Rgba,
    Role,
    SceneGraph,
    StartMeshcat,
)
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    DiscreteContactApproximation,
    MultibodyPlant,
)
from pydrake.solvers import MathematicalProgram, MathematicalProgramResult
from pydrake.systems.framework import DiagramBuilder

from vid2skill.common.drake_envs.drake_gym_wrapper import DrakeGymEnv
from vid2skill.common.helper_functions.misc_helper_functions import (
    get_src_folder_absolute_path,
)


# ---------------- Utility functions to create MultibodyPlant from urdf ---------------- #
class OutputPortType(Enum):
    ACTION = "actions"
    OBSERVATION = "observations"
    REWARD = "rewards"
    TARGET_STATE = "target_state"


def get_quantized_start_time(start_time: float) -> float:
    """Get phase-aligned start time for Drake ``Simulator``.
    As Drake models time stepping as events in a continuous time domain,
    some special care must be taken to ensure each call to
    ``DrakeSystem.step()`` triggers one update. This is done by
    offsetting the simulation duration to advance to ``N * dt + dt/4`` to
    prevent accidentally taking 2 or 0 steps with a call to ``step()``.
    Args:
        start_time: Time step beginning time.
    Returns:
        Time step quantized starting time.
    """
    dt = self.env_configs.sim_dt

    eps = dt / 4

    time_step_phase = start_time % dt
    offset = (dt if time_step_phase > (dt / 2.0) else 0.0) - time_step_phase
    cur_time_quantized = start_time + offset + eps

    return cur_time_quantized


def setup_package_paths(
    parser: Parser, package_map: Dict[str, str], src_folder_abs_path: Path
) -> None:
    """Configure package paths for URDF model parsing.

    Adds package name to path mappings to the parser's package map.
    Paths are resolved relative to the source folder absolute path.

    Args:
        parser (Parser): URDF parser to configure
        package_map (Dict[str, str]): Mapping of package names to relative paths
        src_folder_abs_path (Path): Absolute path to the source folder

    Raises:
        ValueError: If package_map is empty or contains invalid paths
    """
    if not package_map:
        raise ValueError("Package map cannot be empty")

    for name, path in package_map.items():
        parser.package_map().Add(name, (src_folder_abs_path / path).as_posix())


def add_urdf_models(
    parser: Parser, urdfs: List[str], src_folder_abs_path: Path
) -> None:
    """Add URDF models to the parser.

    Loads URDF models from specified file paths, resolving them
    relative to the source folder absolute path.

    Args:
        parser (Parser): URDF parser to add models to
        urdfs (List[str]): List of URDF file paths
        src_folder_abs_path (Path): Absolute path to the source folder

    Raises:
        ValueError: If urdfs list is empty
        FileNotFoundError: If any URDF file cannot be found
    """
    if not urdfs:
        raise ValueError("URDF list cannot be empty")

    for urdf in urdfs:
        full_path = src_folder_abs_path / urdf
        if not full_path.exists():
            raise FileNotFoundError(f"URDF file not found: {full_path}")
        parser.AddModels(full_path.as_posix())


def add_ground_surface(
    plant: MultibodyPlant,
    static_friction_coeff: float = 0.5,
    dynamic_friction_coeff: float = 0.5,
    name: str = "world_ground_plane",
) -> None:
    """Add ground surface to the plant with friction.

    Raises:
        RuntimeError: If plant is already finalized.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by adding ground surface.")

    halfspace_transform = RigidTransform(p=np.array([0, 0, -0.005]))
    friction = CoulombFriction(static_friction_coeff, dynamic_friction_coeff)
    plant.RegisterCollisionGeometry(
        plant.world_body(),
        halfspace_transform,
        Box(10, 10, 0.01),
        name,
        friction,
    )


def weld_links(
    plant: MultibodyPlant, welded_links: Optional[List[Dict[str, str]]]
) -> None:
    """Weld specified links to the world or to each other.

    Args:
        plant (MultibodyPlant): Multibody plant to modify
        welded_links (Optional[List[Dict[str, str]]]): Links to weld, where each dict contains:
            - 'parent_link': Name of the parent link
            - 'body_of_parent_link': Model instance of the parent link
            - 'child_link': Name of the child link to be welded
            - 'body_of_child_link': Model instance of the child link

    Raises:
        RuntimeError: If plant is already finalized.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by welding links.")

    if not welded_links:
        return

    for welded_link in welded_links:
        X_WI = RigidTransform.Identity()
        plant.WeldFrames(
            plant.GetFrameByName(
                welded_link["parent_link"],
                plant.GetModelInstanceByName(welded_link["body_of_parent_link"]),
            ),
            plant.GetFrameByName(
                welded_link["child_link"],
                plant.GetModelInstanceByName(welded_link["body_of_child_link"]),
            ),
            X_WI,
        )


def add_joints(
    plant: MultibodyPlant, joint_specs: Optional[List[Dict[str, Any]]]
) -> None:
    """Add joints to the plant.

    Args:
        plant (MultibodyPlant): Multibody plant to modify
        joint_specs (Optional[List[Dict[str, Any]]]): Joints to add, where each dict contains:
            - 'name': Name of the joint
            - 'parent': Model instance name of the parent body
            - 'child': Model instance name of the child body
            - 'frame_on_parent': Name of the frame on the parent body
            - 'frame_on_child': Name of the frame on the child body
            - 'axis': Rotation axis for the revolute joint
            - 'pos_lower_limit': Lower position limit of the joint
            - 'pos_upper_limit': Upper position limit of the joint
            - 'damping': Damping coefficient for the joint

    Raises:
        RuntimeError: If plant is already finalized.
        KeyError: If required joint configuration keys are missing.
    """
    if plant.is_finalized():
        raise RuntimeError("Cannot modify a finalized plant by adding joints.")

    if not joint_specs:
        return

    for joint in joint_specs:
        # Optional: Add explicit key validation if needed
        required_keys = [
            "name",
            "parent",
            "child",
            "frame_on_parent",
            "frame_on_child",
            "axis",
            "pos_lower_limit",
            "pos_upper_limit",
            "damping",
        ]
        for key in required_keys:
            if key not in joint:
                raise KeyError(f"Missing required joint configuration key: {key}")

        plant.AddJoint(
            RevoluteJoint(
                name=joint["name"],
                frame_on_parent=plant.GetFrameByName(
                    joint["frame_on_parent"],
                    plant.GetModelInstanceByName(joint["parent"]),
                ),
                frame_on_child=plant.GetFrameByName(
                    joint["frame_on_child"],
                    plant.GetModelInstanceByName(joint["child"]),
                ),
                axis=joint["axis"],
                pos_lower_limit=joint["pos_lower_limit"],
                pos_upper_limit=joint["pos_upper_limit"],
                damping=joint["damping"],
            )
        )


def create_plant_from_urdfs(
    builder: DiagramBuilder,
    package_map: Dict[str, str],
    urdfs: List[str],
    sim_dt: float,
    welded_links: Optional[List[Dict[str, str]]] = None,
    joint_specs: Optional[List[Dict[str, Any]]] = None,
    meshcat: Optional[Meshcat] = None,
    show_collision: bool = False,
    visualizer_publish_period: float = -1,
) -> Tuple[MultibodyPlant, SceneGraph, MeshcatVisualizer, Optional[MeshcatVisualizer]]:
    """Create a multibody plant with specified URDF models and configurations.

    Args:
        builder (DiagramBuilder): Diagram builder for plant creation
        package_map (Dict[str, str]): Package name to path mapping
        urdfs (List[str]): List of URDF file paths
        sim_dt (float): Simulation time step
        welded_links (Optional[List[Dict[str, str]]], optional): Links to weld. Defaults to None.
        joint_specs (Optional[List[Dict[str, Any]]], optional): Specs of joints to add. Defaults to None.
        meshcat (Optional[Meshcat], optional): Meshcat instance. Defaults to None.
        show_collision (bool, optional): Show collision geometry. Defaults to False.
        visualizer_publish_period (float, optional): Visualizer publish period. Defaults to -1.

    Returns:
        Tuple containing MultibodyPlant, SceneGraph, MeshcatVisualizer, and optional MeshcatVisualizer
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, sim_dt)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)

    parser = Parser(plant)
    parser.SetAutoRenaming(True)

    src_folder_abs_path = get_src_folder_absolute_path()
    # setup_package_paths(parser, package_map, src_folder_abs_path)
    add_urdf_models(parser, urdfs, src_folder_abs_path)

    # perform various operations on the plant
    weld_links(plant, welded_links)
    add_ground_surface(plant)
    add_joints(plant, joint_specs)

    plant.Finalize()

    visual_visualizer = None
    collision_visualizer = None

    if meshcat is not None:
        visual_visualizer, collision_visualizer = setup_drake_visualizers(
            builder, scene_graph, meshcat, show_collision, visualizer_publish_period
        )
    return plant, scene_graph, visual_visualizer, collision_visualizer


# ------------------------------------------------------------------------------------#


# ---------------- Utility functions to set up visualizers in Drake ---------------- #
def initialize_meshcat() -> Meshcat:
    """Initialize and prepare Meshcat visualizer.

    Returns:
        Configured Meshcat instance
    """
    meshcat = StartMeshcat()
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    return meshcat


def set_camera_position_in_meshcat(
    meshcat: Meshcat,
    camera_position: Optional[npt.NDArray[np.float64]] = None,
    camera_target: Optional[npt.NDArray[np.float64]] = None,
) -> None:
    """Configure default camera position.

    Args:
        meshcat (Meshcat): Meshcat visualizer to configure
        camera_position (npt.NDArray[np.float64]): Camera position in world coordinates
        camera_target (npt.NDArray[np.float64]): Camera target in world coordinates
    """
    if camera_position is None:
        camera_position = np.array([0.4, 0.4, 0.4])
    if camera_target is None:
        camera_target = np.zeros(3)
    meshcat.SetCameraPose(camera_position, camera_target)


def add_meshcat_visualizer(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    meshcat_visualizer_params: MeshcatVisualizerParams,
) -> MeshcatVisualizer:
    """Add visual visualizer to the diagram builder.

    Args:
        builder (DiagramBuilder): Diagram builder
        scene_graph (SceneGraph): Scene graph to visualize
        meshcat (Meshcat): Meshcat instance
        meshcat_visualizer_params (MeshcatVisualizerParams): Visualizer parameters

    Returns:
        Created MeshcatVisualizer
    """
    if builder.already_built():
        raise RuntimeError("Cannot add visualizer to an already built diagram")

    return MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, meshcat_visualizer_params
    )


def setup_drake_visualizers(
    builder: DiagramBuilder,
    scene_graph: SceneGraph,
    meshcat: Meshcat,
    show_collision: bool = False,
    publish_period: float = -1,
) -> Tuple[MeshcatVisualizer, Optional[MeshcatVisualizer]]:
    """Set up Drake visualizer if enabled.

    Args:
        builder (DiagramBuilder): Diagram builder
        scene_graph (SceneGraph): Scene graph
        meshcat (Meshcat): Meshcat instance
        show_collision (bool): Show collision geometry
        publish_period (float): Period to publish visualizer updates. Defaults to -1.

    Returns:
        (MeshcatVisualizer, MeshcatVisualizer | None)
    """
    if publish_period == -1:
        raise RuntimeError("publish_period must be > 0")
    set_camera_position_in_meshcat(meshcat)

    collision_visualizer = None

    visual_params = MeshcatVisualizerParams(
        role=Role.kPerception, prefix="visual", publish_period=publish_period
    )
    collision_params = MeshcatVisualizerParams(
        role=Role.kProximity, prefix="collision", publish_period=publish_period
    )
    visual_visualizer = add_meshcat_visualizer(
        builder, scene_graph, meshcat, visual_params
    )

    if show_collision:
        collision_visualizer = add_meshcat_visualizer(
            builder, scene_graph, meshcat, collision_params
        )

    return visual_visualizer, collision_visualizer


def draw_frame_axes(
    meshcat: Meshcat,
    str_path: str,
    frame_quat: Optional[np.ndarray] = None,
    frame_pos: Optional[np.ndarray] = None,
    transparency: float = 1.0,
) -> None:
    if not meshcat.HasPath(f"{str_path}/x_axis"):
        meshcat.SetObject(
            f"{str_path}/x_axis",
            Cylinder(0.001, 0.05),
            Rgba(1.0, 0.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/y_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 1.0, 0.0, transparency),
        )
        meshcat.SetObject(
            f"{str_path}/z_axis",
            Cylinder(0.001, 0.05),
            Rgba(0.0, 0.0, 1.0, transparency),
        )
        x_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 1, 0])), np.array([0.025, 0, 0])
        )
        y_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([1, 0, 0])), np.array([0, 0.025, 0])
        )
        z_axis_transform = RigidTransform(
            AngleAxis(np.pi / 2, np.array([0, 0, 1])), np.array([0, 0, 0.025])
        )
        meshcat.SetTransform(f"{str_path}/x_axis", x_axis_transform)
        meshcat.SetTransform(f"{str_path}/y_axis", y_axis_transform)
        meshcat.SetTransform(f"{str_path}/z_axis", z_axis_transform)

    if frame_quat is not None and frame_pos is not None:
        meshcat.SetTransform(
            str_path,
            RigidTransform(quaternion=Quaternion(frame_quat), p=frame_pos),
        )


def visualize_traj_with_meshcat(drake_system, x_traj, timestep=0.04):
    visualizer_context = drake_system.visual_visualizer.GetMyContextFromRoot(
        drake_system.diagram_context
    )
    drake_system.visual_visualizer.StartRecording()
    traj_length = x_traj.shape[0]
    for i in range(traj_length):
        start_time = time.perf_counter()
        drake_system.diagram_context.SetTime(i * timestep)
        drake_system.plant.SetPositionsAndVelocities(
            drake_system.plant_context, x_traj[i]
        )
        drake_system.visual_visualizer.ForcedPublish(visualizer_context)

        time_remaining = timestep - (time.perf_counter() - start_time)
        if time_remaining > 0:
            time.sleep(time_remaining)

    drake_system.visual_visualizer.StopRecording()
    animation = drake_system.visual_visualizer.get_mutable_recording()
    drake_system.meshcat.SetAnimation(animation)


# ------------------------------------------------------------------------------------#


# ------------------------ Utility functions for drake-gym -------------------------- #
def make_env(cfg: DictConfig):
    """Create a gym-like wrapper around DrakeSystem instance from the provided configuration."""
    drake_system = hydra.utils.instantiate(cfg)

    action_space = gym.spaces.Box(
        low=drake_system.sys_configs.min_allowable_acts,
        high=drake_system.sys_configs.max_allowable_acts,
        dtype=np.float64,
    )

    observation_space = gym.spaces.Box(
        low=drake_system.sys_configs.terminated_min_obs,
        high=drake_system.sys_configs.terminated_max_obs,
        dtype=np.float64,
    )

    return DrakeGymEnv(
        drake_system=drake_system,
        simulator=drake_system.sim,
        time_step=drake_system.sys_configs.sim_dt * drake_system.sys_configs.frame_skip,
        high_level_action_space=action_space,
        high_level_observation_space=observation_space,
        reward=OutputPortType.REWARD.value,
        action_port_id=OutputPortType.ACTION.value,
        target_state_port_id=None,
        observation_port_id=OutputPortType.OBSERVATION.value,
        get_initial_state_fn=drake_system.get_default_initial_state,
        state_size=drake_system.state_size,
    )


def setup_environment(cfg: DictConfig) -> Tuple[object, np.ndarray]:
    """
    Initialize the environment and get initial state.

    Args:
        cfg: Configuration dictionary

    Returns:
        Tuple containing environment object and initial state
    """
    env = make_env(cfg)
    env.reset()
    return env


# ------------------------------------------------------------------------------------#


# --------------------- Utility functions for MathematicalProgram ------------------- #
def convert_mathematical_program_result_to_dict(
    result: MathematicalProgramResult, prog: MathematicalProgram
) -> Dict[str, Any]:
    """Converts a MathematicalProgramResult to a dictionary.

    Args:
        result: The result of solving a MathematicalProgram.
        prog: The corresponding MathematicalProgram.

    Returns:
        A dictionary containing the optimization results.
    """
    return {
        "decision_variables": {
            str(var): result.GetSolution(var) for var in prog.decision_variables()
        },
        "optimal_cost": result.get_optimal_cost(),
        "solver_id": str(result.get_solver_id().name()),
        "is_success": result.is_success(),
        "status": result.get_solution_result().name,
    }


# ------------------------------------------------------------------------------------#
