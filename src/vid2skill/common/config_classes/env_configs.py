"""Data classes which are used for configuring Drake environments."""

import collections
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Tuple, List
from scipy.spatial.transform import Rotation


@dataclass
class BaseDrakeSystemConfigs:
    """Store config parameters for building Drake systems."""

    state_size: int = -1  # must be overridden by subclasses
    action_size: int = -1  # must be overridden by subclasses
    frame_skip: int = 1
    sim_dt: float = 0.002
    show_collision_geometries: bool = False
    enable_contact_visualizer: bool = False

    package_map: Dict[str, str] = field(default=dict)
    urdfs: List[str] = field(default=list)
    welded_links: Optional[str] = None
    additional_joints: Optional[Tuple[Dict[str, str]]] = None


@dataclass
class FrankaAllegroDrakeSystemConfigs(BaseDrakeSystemConfigs):
    # target position of the object
    target_pos_obj: Sequence[float] = field(
        default_factory=lambda: np.array([0.0, 0.05, 0.1])
    )

    # target orientation of the object
    target_axisangle_obj: Sequence[float] = field(
        default_factory=lambda: ([0.0, 0.0, 1.0, 1.5])
    )  # [axis, angle]

    # initial position of the object
    init_pos_obj: Sequence[float] = field(default_factory=lambda: ([0.3, 0.3, 0.05]))

    # initial orientation of the object with axis-angle representation [axis, angle]
    init_axisangle_obj: Sequence[float] = field(
        default_factory=lambda: ([0.0, 0.0, 1.0, 0.0])
    )

    # initial velocity of the object
    init_vel_obj: Sequence[float] = field(
        default_factory=lambda: ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    # initial joint positions and velocities of the franka robot
    init_qpos_franka: Sequence[float] = field(default_factory=lambda: ([]))
    init_qvel_franka: Sequence[float] = field(default_factory=lambda: ([0.0] * 9))

    # initial joint positions and velocities of the Allegro hand
    init_qpos_allegro: Sequence[float] = field(default_factory=lambda: ([0.0] * 16))
    init_qvel_allegro: Sequence[float] = field(default_factory=lambda: ([0.0] * 16))

    # for lcs approximation
    contact_geoms: Sequence[Tuple[str, str]] = field(default_factory=lambda: ([]))

    def __post_init__(self):
        # target position and orientation of the object
        self.target_pos_obj = np.array(self.target_pos_obj)
        self.target_axisangle_obj = np.array(self.target_axisangle_obj)
        self.target_quat_obj = Rotation.from_rotvec(
            self.target_axisangle_obj[:3] * self.target_axisangle_obj[3]
        ).as_quat(scalar_first=True)

        # initial position and orientation of the object
        self.init_pos_obj = np.array(self.init_pos_obj)
        self.init_axisangle_obj = np.array(self.init_axisangle_obj)
        self.init_quat_obj = Rotation.from_rotvec(
            self.init_axisangle_obj[:3] * self.init_axisangle_obj[3]
        ).as_quat(scalar_first=True)

        self.init_qpos_obj = np.concatenate((self.init_quat_obj, self.init_pos_obj))
        self.init_qvel_obj = np.array(self.init_vel_obj)

        # initial joint positions and velocities of the franka_allegro
        self.init_qpos_franka = np.array(self.init_qpos_franka)
        self.init_qvel_franka = np.array(self.init_qvel_franka)
        self.init_qpos_allegro = np.array(self.init_qpos_allegro)
        self.init_qvel_allegro = np.array(self.init_qvel_allegro)

        self.init_qpos = np.concatenate(
            (self.init_qpos_franka, self.init_qpos_allegro, self.init_qpos_obj)
        )
        self.init_qvel = np.concatenate(
            (self.init_qvel_franka, self.init_qvel_allegro, self.init_qvel_obj)
        )
        self.init_state = np.concatenate((self.init_qpos, self.init_qvel))

        # lower/upper bounds for control inputs
        self.min_allowable_acts = np.array([-20.0] * self.action_size)
        self.max_allowable_acts = np.array([20.0] * self.action_size)
        self.min_allowable_obs = np.array([-200.0] * self.state_size)
        self.max_allowable_obs = np.array([200.0] * self.state_size)
        self.terminated_min_obs = np.array(self.min_allowable_obs)
        self.terminated_max_obs = np.array(self.max_allowable_obs)
