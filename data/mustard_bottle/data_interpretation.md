# Directory Structure
In DexYCB dataset, there are 10 subjects. Each subject has 20 objects. Each object video sequences are split into 5 folders. Each folder has 8 sub-folders which corresponding to 8 different camera setups. I have extract the Mustard Bottle object related folders out.

# Data Sturcture:
Each `label_xxxxxx.npz` file contains the following annotations packed in a dictionary:

- `seg`: A unit8 numpy array of shape [H, W] containing the segmentation map. The label of each pixel can be 0 (background), 1-21 (YCB object), or 255 (hand).
- `pose_y`: A float32 numpy array of shape [num_obj, 3, 4] holding the 6D pose of each object in the camera coordinates. Each 6D pose is represented by [R; t], where R is the 3x3 rotation matrix and t is the 3x1 translation.
- `pose_m`: A float32 numpy array of shape [1, 51] holding the pose of the hand in the camera coordinates. pose_m[:, 0:48] stores the MANO pose coefficients in PCA representation, and pose_m[0, 48:51] stores the translation. If the image does not have a visible hand or the annotation does not exist, pose_m will be all 0.
- `joint_3d`: A float32 numpy array of shape [1, 21, 3] holding the 3D joint position of the hand in the camera coordinates. The joint order is specified here. If the image does not have a visible hand or the annotation does not exist, joint_3d will be all -1.
- `joint_2d`: A float32 numpy array of shape [1, 21, 2] holding the 2D joint position of the hand in the image space. The joint order follows joint_3d. If the image does not have a visible hand or the annotation does not exist, joint_2d will be all -1.


# Order of hand joints
DexYCB dataset uses the convention from MANO project to define the order of hand joints.

0. wrist
1. thumb_mcp
2. thumb_pip
3. thumb_dip
4. thumb_tip
5. index_mcp
6. index_pip
7. index_dip
8. index_tip
9. middle_mcp
10. middle_pip
11. middle_dip
12. middle_tip
13. ring_mcp
14. ring_pip
15. ring_dip
16. ring_tip
17. little_mcp
18. little_pip
19. little_dip
20. little_tip

# Camera calibration:

We store extrinsics for all cameras and AprilTag in the `extrinsics.yml` file of the `camera_calibration` folder.

Each extrinsics matrix is the transformation from the camera coordinates to the master camera coordinates. Each dataset might have different master camera.

The world frame is defined by an AprilTag fixed at one table's corner in the scene. By inverting the extrinsics matrix of the apriltag, we can get the transformation from the master camera coordinates to the world coordinates.
