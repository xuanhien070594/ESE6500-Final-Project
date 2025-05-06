from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from pydrake.common.value import AbstractValue
from pydrake.common.eigen_geometry import AngleAxis, Quaternion
from pydrake.geometry import Box, Cylinder, GeometryId, Rgba, Sphere, Mesh
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

        self.hand_body_names = [
            "panda_link8",
            "allegro_link_13",
            "allegro_link_14",
            "allegro_link_15",
            "allegro_link_15_tip",
            "allegro_link_1",
            "allegro_link_2",
            "allegro_link_3",
            "allegro_link_3_tip",
            "allegro_link_5",
            "allegro_link_6",
            "allegro_link_7",
            "allegro_link_7_tip",
            "allegro_link_9",
            "allegro_link_10",
            "allegro_link_11",
            "allegro_link_11_tip",
        ]
        self.franka_allegro_model_instance = self.plant.GetModelInstanceByName(
            "franka_allegro"
        )
        self.mustard_bottle_model_instance = self.plant.GetModelInstanceByName(
            "006_mustard_bottle"
        )

    def visualize_ref_obj_pose(self, obj_pose: np.ndarray):
        ref_obj_path = "/ref_obj/"
        geom = Mesh(
            "/home/hienbui/git/ESE6500-Final-Project/src/vid2skill/models/ycb/meshes/006_mustard_bottle_textured.gltf",
        )
        if not self.meshcat.HasPath(ref_obj_path):
            self.meshcat.SetObject(
                ref_obj_path,
                geom,
            )
        self.meshcat.SetTransform(
            ref_obj_path,
            RigidTransform(R=RotationMatrix(obj_pose[:3, :3]), p=obj_pose[:3, 3]),
        )

    def visualize_hand_pose(self, hand_pose: np.ndarray, is_human_hand: bool = True):
        ref_hand_path = "/ref_hand/" if is_human_hand else "/hand/"
        n_hand_joints = 21 if is_human_hand else 17
        transparency = 0.2
        wrist_geom = Box(0.03, 0.03, 0.03)
        joint_geom = Box(0.01, 0.01, 0.01)
        color = (
            Rgba(1.0, 0, 0, transparency)
            if is_human_hand
            else Rgba(0, 1.0, 1.0, transparency)
        )

        if is_human_hand:
            fingers = {
                "thumb_finger_joint_indices": [0, 1, 2, 3, 4],
                "index_finger_joint_indices": [0, 5, 6, 7, 8],
                "middle_finger_joint_indices": [0, 9, 10, 11, 12],
                "ring_finger_joint_indices": [0, 13, 14, 15, 16],
                "little_finger_joint_indices": [0, 17, 18, 19, 20],
            }
        else:
            fingers = {
                "thumb_finger_joint_indices": [0, 1, 2, 3, 4],
                "index_finger_joint_indices": [0, 5, 6, 7, 8],
                "middle_finger_joint_indices": [0, 9, 10, 11, 12],
                "little_finger_joint_indices": [0, 13, 14, 15, 16],
            }

        # Draw each joint of the hand as a small cube
        for i in range(n_hand_joints):
            joint_path = f"{ref_hand_path}/joint_{i}"
            if not self.meshcat.HasPath(joint_path):
                geom = wrist_geom if i == 0 else joint_geom
                self.meshcat.SetObject(joint_path, geom, color)
            self.meshcat.SetTransform(
                joint_path,
                RigidTransform(
                    p=hand_pose[i],
                ),
            )

        # Draw bones between joints
        for finger in fingers:
            joint_indices = fingers[finger]
            for j in range(len(joint_indices) - 1):
                joint_1_index = joint_indices[j]
                joint_2_index = joint_indices[j + 1]
                joint_1_pos = hand_pose[joint_1_index]
                joint_2_pos = hand_pose[joint_2_index]

                # Calculate the midpoint and the rotation axis
                mid_point = (joint_1_pos + joint_2_pos) / 2
                direction = joint_2_pos - joint_1_pos
                length = np.linalg.norm(direction)
                direction /= length

                # Create a cylinder to represent the bone
                bone_path = f"{ref_hand_path}/bone_{joint_1_index}_{joint_2_index}"
                geom = Cylinder(0.003, length)
                self.meshcat.SetObject(bone_path, geom, color)
                self.meshcat.SetTransform(
                    bone_path,
                    RigidTransform(
                        RotationMatrix.MakeFromOneVector(direction, 2),
                        mid_point,
                    ),
                )

    @lru_cache(maxsize=None)
    def get_contact_pairs(
        self,
    ) -> Tuple[List[Tuple[GeometryId, GeometryId]], List[Tuple[RigidBody, RigidBody]]]:
        contact_pairs = []
        body_pairs = []
        query_port = self.plant.get_geometry_query_input_port()
        query_object = query_port.Eval(self.plant_context)
        inspector = query_object.inspector()
        for geom_pair in self.sys_configs.contact_geoms:
            body_1 = self.plant.GetBodyByName(
                geom_pair["body_1"],
                self.plant.GetModelInstanceByName(geom_pair["model_1"]),
            )
            body_2 = self.plant.GetBodyByName(
                geom_pair["body_2"],
                self.plant.GetModelInstanceByName(geom_pair["model_2"]),
            )
            geom_ids_for_body_1 = self.plant.GetCollisionGeometriesForBody(body_1)
            geom_ids_for_body_2 = self.plant.GetCollisionGeometriesForBody(body_2)

            geom_1_id = None
            geom_2_id = None
            for geom_id in geom_ids_for_body_1:
                if geom_pair["collision_1"] in inspector.GetName(geom_id):
                    geom_1_id = geom_id
                    break

            for geom_id in geom_ids_for_body_2:
                if geom_pair["collision_2"] in inspector.GetName(geom_id):
                    geom_2_id = geom_id
                    break

            if geom_1_id is None:
                raise ValueError(
                    f"Geometry {geom_pair['collision_1']} not found for body {body_1.name()}"
                )
            if geom_2_id is None:
                raise ValueError(
                    f"Geometry {geom_pair['collision_2']} not found for body {body_2.name()}"
                )

            contact_pairs.append(
                (
                    geom_1_id,
                    geom_2_id,
                )
            )
            body_pairs.append((body_1, body_2))
        return contact_pairs, body_pairs

    @property
    def n_contacts_lcs_approx(self):
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
