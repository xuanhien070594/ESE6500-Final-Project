import numpy as np
from pathlib import Path
import hydra
import ipdb
import yaml
import time
from omegaconf import DictConfig
from vid2skill.common.helper_functions.drake_helper_functions import make_env


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
    ipdb.set_trace()

    for i in range(ref_hand_poses.shape[0]):
        env.drake_system.visualize_ref_hand_pose(ref_hand_poses[i])
        env.drake_system.visualize_ref_obj_pose(ref_obj_poses[i])
        time.sleep(0.1)


def load_dataset(name, camera_id):
    data_folder = Path(__file__).parent.parent / "data/mustard_bottle/"
    data_folder_specific_view = data_folder / name / camera_id
    extrinsics_path = data_folder / "camera_calibration" / f"extrinsics.yml"

    # Load the extrinsics matrices of all cameras in the setup.
    # - Each extrinsics matrix is the transformation from the camera frame to the master camera frame
    #   in this dataset, the master camera is the one with id 840412060917.
    # - The world frame is defined by the apriltag in the scene. By inverting the extrinsics matrix of the
    #   apriltag, we can get the transformation from the world frame to the master camera frame.
    with open(extrinsics_path, "r") as f:
        extrinsics = yaml.safe_load(f)
    masterTcamera = np.eye(4)
    masterTcamera[:3, :] = np.array(extrinsics["extrinsics"][camera_id]).reshape((3, 4))

    masterTworld = np.eye(4)
    masterTworld[:3, :] = np.array(extrinsics["extrinsics"]["apriltag"]).reshape((3, 4))
    worldTmaster = np.linalg.inv(masterTworld)

    worldTcamera = worldTmaster @ masterTcamera

    # The world frame defined in the dataset is different from the one in Drake
    # we need to transform the world frame to the Drake world frame
    offset = np.eye(4)
    offset[:3, :3] = np.array(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ]
    )
    offset[:3, 3] = np.array([0.7, -0.8, 0.0])
    worldTcamera = offset @ worldTcamera

    obj_poses = []
    hand_poses = []
    obj_index = 0  # the index for mustard bottle

    # Each label.npz file contains the pose of the object and the hand wrt the camera
    # coordinates for each frame
    for file in sorted(data_folder_specific_view.glob("labels_*.npz")):
        data = np.load(file)
        obj_pose_in_cam_frame = np.eye(4)
        obj_pose_in_cam_frame[:3, :3] = data["pose_y"][obj_index, :, :3]
        obj_pose_in_cam_frame[:3, 3] = data["pose_y"][obj_index, :, 3]

        offset_obj_pose = np.array([0.0266, 0, -0.09315])
        obj_pose_in_cam_frame[:3, 3] += obj_pose_in_cam_frame[:3, :3] @ offset_obj_pose
        obj_pose_in_world_frame = worldTcamera @ obj_pose_in_cam_frame
        obj_poses.append(obj_pose_in_world_frame)

        hand_pose = data["joint_3d"] @ worldTcamera[:3, :3].T + worldTcamera[:3, 3]
        hand_poses.append(hand_pose)
    obj_poses = np.array(obj_poses)
    print(obj_poses[0])
    hand_poses = np.concatenate(hand_poses, axis=0)
    return obj_poses, hand_poses


if __name__ == "__main__":
    main()
