# ESE6500-Final-Project
This project proposes an imitation learning (IL)
framework leveraging human demonstrations to enable data-
efficient and realistic dexterous manipulation. Specifically, we
will extract trajectories from human videos, incorporate phys-
ical constraints to ensure realistic interactions, and apply
trajectory optimization techniques to generate dynamically
feasible robot motions. The final policy will be trained using the
optimized trajectories within a Diffusion Policy framework. The
expected outcomes include a significant reduction in training
data requirements compared to traditional methods and more
realistic policy generation compared to existing approaches.

![Overall Framework](media/vid2skill.drawio.png)

*Figure 1: Video-to-Skill Framework Overview: We begin by extracting hand and object poses from videos, followed by kinematic motion retargeting to generate kinematically feasible trajectories for the Franka-Allegro system. These trajectories are then refined using trajectory optimization, which incorporates contact and dynamic constraints to ensure dynamic feasibility. Next, we introduce local perturbations to physical parameters, such as the object’s initial state or friction coefficients. For each perturbed setting, we re-solve the trajectory optimization to adapt to the new conditions. After this process, we obtain a diverse set of physically plausible trajectories that can be used as training data for Diffusion Policy.*

![Kinematic Retargeting Trajectory](media/kinematic_retargeting.drawio.png)

*Figure 2: Motion tiles illustrating the reference trajectory extracted from the video (top row) and the trajectory produced via kinematic retargeting (bottom row) for the grasping-mustard-bottle task.*

## Installation Instructions

1. Install uv (Python package installer):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create virtual environment and install project dependencies:
```bash
uv venv
uv pip install -e .
```

## Running the Environment Test

To run the Franka Allegro environment test:
```bash
uv run python scripts/drake_env_test.py
```

## Running Kinematic Motion Retargeting

```bash
uv run python scripts/kinematic_retargeting_test.py
```

## Miscellaneous

### How to check the urdf model

```bash
uv run python -m pydrake.visualization.model_visualizer <path-to-urdf-file>
```

## References

[1] Z. Chen, S. Chen, E. Arlaud, I. Laptev, and C. Schmid, “ViViDex:
Learning vision-based dexterous manipulation from human videos,”
arXiv preprint arXiv:2404.15709, 2024.

[2] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake,
and S. Song, “Diffusion policy: Visuomotor policy learning via action
diffusion,” The International Journal of Robotics Research, 2024.

[3] L. Yang, H. J. T. Suh, T. Zhao, B. P. Graesdal, T. Kelestemur,
J. Wang, T. Pang, and R. Tedrake, “Physics-driven data generation fo
manipulation via trajectory,” arXiv preprint arXiv:2502.20382, 2025,
[Online]. Available: https://arxiv.org/abs/2502.20382.