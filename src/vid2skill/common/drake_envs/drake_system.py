import numpy as np

from typing import Type, Tuple
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    DiagramBuilder,
    LeafSystem,
    Diagram,
    Context,
    Context_,
)
from pydrake.multibody.plant import MultibodyPlant_, MultibodyPlant
from pydrake.autodiffutils import AutoDiffXd

from vid2skill.common.config_classes.env_configs import BaseDrakeSystemConfigs
from vid2skill.common.helper_functions.drake_helper_functions import (
    create_plant_from_urdfs,
    initialize_meshcat,
)


class BaseDrakeSystem:
    def __init__(self, drake_sys_configs: Type[BaseDrakeSystemConfigs]) -> None:
        """Instantiate DrakeSystems object given a config object

        Args:
            drake_sys_configs (Type[DrakeSystemConfigs]): object stores the configuration of DrakeSystems
        """
        self.sys_configs = drake_sys_configs
        self.diagram = None
        self.sim = None

        self.builder = DiagramBuilder()

        # shared Meshcat instance for all visualizers
        self.meshcat = initialize_meshcat()
        (
            self.plant,
            self.scene_graph,
            self.visual_visualizer,
            self.collision_visualizer,
        ) = create_plant_from_urdfs(
            builder=self.builder,
            package_map=self.sys_configs.package_map,
            urdfs=self.sys_configs.urdfs,
            sim_dt=self.sys_configs.sim_dt,
            welded_links=self.sys_configs.welded_links,
            joint_specs=self.sys_configs.additional_joints,
            meshcat=self.meshcat,
            show_collision=self.sys_configs.show_collision_geometries,
            visualizer_publish_period=(
                self.sys_configs.sim_dt * self.sys_configs.frame_skip
            ),
        )

        # check if the state and input size of the plant is consistent with the config
        assert self.plant.num_multibody_states() == self.sys_configs.state_size
        assert self.plant.num_actuators() == self.sys_configs.action_size

    def build_diagram(self):
        """Build diagram and use it to initialize the Drake simulator."""
        assert not self.is_diagram_built, (
            "The diagram is already built, the second call of this function is not"
            " allowed."
        )
        self.diagram = self.builder.Build()
        self.diagram_context = self.diagram.CreateDefaultContext()

        self.sim = Simulator(self.diagram)
        self.sim.Initialize()
        self.sim.set_publish_every_time_step(False)

    def set_state(self, initial_state: np.ndarray) -> None:
        assert self.is_diagram_built, "Please call function to build the diagram first."
        assert initial_state.shape[0] == self.state_size, (
            f"Invalid state size, expected {self.state_size} but got"
            f" {initial_state.shape[0]}"
        )
        self.plant.SetPositionsAndVelocities(self.plant_context, initial_state)
        self.sim.Initialize()

    def get_state(self) -> np.ndarray:
        """Retrieve the current state of the system."""
        assert self.is_diagram_built, "Please call function to build the diagram first."
        return self.plant.GetPositionsAndVelocities(self.plant_context)

    def get_default_initial_state(self) -> np.ndarray:
        raise NotImplementedError("Child classes should implement this method.")

    def design_reward_system(self) -> LeafSystem:
        raise NotImplementedError("Child classes should implement this method.")

    @property
    def is_diagram_built(self) -> bool:
        """Flag to check if the diagram has been built."""
        return self.diagram is not None and self.sim is not None

    @property
    def plant_context(self):
        """Return mutable context of the plant (where stores all simulation data)."""
        return self.plant.GetMyMutableContextFromRoot(self.diagram_context)

    @property
    def state_size(self):
        """Return dimension of task environment."""
        return self.plant.num_multibody_states()

    @property
    def action_size(self):
        """Return dimension of action."""
        return self.plant.num_actuators()
