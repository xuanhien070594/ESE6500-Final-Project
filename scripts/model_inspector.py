import os

import click
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    StartMeshcat,
    Meshcat,
)
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder


def model_inspector(meshcat: Meshcat, filename: str):
    meshcat.Delete()
    meshcat.DeleteAddedControls()
    builder = DiagramBuilder()

    # Note: the time_step here is chosen arbitrarily.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    # Load the file into the plant/scene_graph.
    parser = Parser(plant)
    parser.AddModels(filename)
    plant.Finalize()

    # Add two visualizers, one to publish the "visual" geometry, and one to
    # publish the "collision" geometry.
    visual = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"),
    )
    collision = MeshcatVisualizer.AddToBuilder(
        builder,
        scene_graph,
        meshcat,
        MeshcatVisualizerParams(role=Role.kProximity, prefix="collision"),
    )
    # Disable the collision geometry at the start; it can be enabled by the
    # checkbox in the meshcat controls.
    meshcat.SetProperty("collision", "visible", False)

    # Set the timeout to a small number in test mode. Otherwise, JointSliders
    # will run until "Stop JointSliders" button is clicked.
    default_interactive_timeout = 1.0 if "TEST_SRCDIR" in os.environ else None
    sliders = builder.AddSystem(JointSliders(meshcat, plant))
    diagram = builder.Build()
    sliders.Run(diagram, default_interactive_timeout)


@click.command
@click.option(
    "--env_folder",
    type=str,
    default="sphere_fingertip_with_cube",
    help="The name of the environment folder",
)
@click.option(
    "--urdf_file",
    type=str,
    default="square_frame_planar",
    help="The name of the urdf file",
)
def main(env_folder, urdf_file):
    meshcat = StartMeshcat()
    filename = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            f"../src/vid2skill/models/{env_folder}/{urdf_file}.urdf",
        )
    )
    model_inspector(meshcat, filename)


if __name__ == "__main__":
    main()
