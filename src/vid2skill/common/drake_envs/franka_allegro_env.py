from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import Box, Cylinder, GeometryId, Rgba, Sphere
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.tree import RigidBody
from pydrake.systems.framework import LeafSystem
from pydrake.systems.primitives import PassThrough, ConstantVectorSource
from scipy.spatial.transform import Rotation

from vid2skill.common.helper_functions.drake_helper_functions import (
    OutputPortType,
    draw_frame_axes,
)
from vid2skill.common.config_classes.env_configs import FrankaAllegroDrakeSystemConfigs
from vid2skill.common.drake_envs.drake_system import BaseDrakeSystem


class FrankaAllegroDrakeSystem(BaseDrakeSystem):
    def __init__(self, configs: FrankaAllegroDrakeSystemConfigs) -> None:
        super().__init__(drake_sys_configs=configs)
        desired_action_source = self.builder.AddSystem(
            PassThrough(vector_size=self.plant.num_actuators())
        )
        fake_reward_source = self.builder.AddSystem(ConstantVectorSource(np.zeros(1)))

        self.builder.Connect(
            desired_action_source.get_output_port(),
            self.plant.get_actuation_input_port(),
        )

        self.builder.ExportInput(
            desired_action_source.get_input_port(), OutputPortType.ACTION.value
        )
        self.builder.ExportOutput(
            self.plant.get_state_output_port(), OutputPortType.OBSERVATION.value
        )
        self.builder.ExportOutput(
            fake_reward_source.get_output_port(), OutputPortType.REWARD.value
        )
        self.build_diagram()

    @lru_cache(maxsize=None)
    def get_contact_pairs(
        self,
    ) -> Tuple[List[Tuple[GeometryId, GeometryId]], List[Tuple[RigidBody, RigidBody]]]:
        assert hasattr(self, "plant_lcs"), "No plant for LCS approximation was built."
        contact_pairs = []
        body_pairs = []
        for geom_pair in self.sys_configs.contact_geoms:
            body_1 = self.plant_lcs.GetBodyByName(
                geom_pair["link_1"],
                self.plant_lcs.GetModelInstanceByName(geom_pair["body_1"]),
            )
            body_2 = self.plant_lcs.GetBodyByName(
                geom_pair["link_2"],
                self.plant_lcs.GetModelInstanceByName(geom_pair["body_2"]),
            )
            geom_1_id = self.plant_lcs.GetCollisionGeometriesForBody(body_1)[0]
            geom_2_id = self.plant_lcs.GetCollisionGeometriesForBody(body_2)[0]
            contact_pairs.append(
                (
                    geom_1_id,
                    geom_2_id,
                    geom_pair["align_tangential_force_basis_to_linear_vel"],
                )
            )
            body_pairs.append((body_1, body_2))
        return contact_pairs, body_pairs

    @property
    def n_contacts_lcs_approx(self):
        print(self.sys_configs.contact_geoms)
        return len(self.sys_configs.contact_geoms)

    def get_default_initial_state(
        self,
    ) -> np.ndarray:  # TODO: change the name of this function to `reset`
        """Add noise to the initial system state."""
        return self.sys_configs.init_state

    def visualize_target_state_and_workspace_limits(self):
        draw_frame_axes(
            self.meshcat,
            str_path="/target/",
            frame_quat=self.sys_configs.target_quat_obj,
            frame_pos=self.sys_configs.target_pos_obj,
        )

    def get_target_states(self):
        target_states = np.zeros((2, self.state_size))

        target_states[1, 9:13] = self.sys_configs.target_quat_obj
        target_states[1, 13:16] = self.sys_configs.target_pos_obj
        return target_states

    def set_target_state(self, new_target_state: np.ndarray):
        assert new_target_state.shape == (self.state_size,)
        self.sys_configs.target_quat_obj = new_target_state[9:13]
        self.sys_configs.target_pos_obj = new_target_state[13:16]
