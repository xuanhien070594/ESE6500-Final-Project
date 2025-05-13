# ESE6500-Final-Project

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