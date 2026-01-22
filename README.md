# LeRobot Policy A2

A2 Policy package for [LeRobot](https://github.com/huggingface/lerobot) - ViLGP3D grasp and place policy using CLIP-based action selection for tabletop manipulation with UR5e robot.

## Overview

This package implements the ViLGP3D policy for 6-DOF grasp and place pose selection in tabletop manipulation tasks. It uses:

- **CLIP** for visual feature extraction from RGB images
- **Cross-attention transformer** for fusing point cloud and action features
- **Rotary Position Encoding (RoPE)** for 3D positional information
- **GraspNet** for candidate grasp pose generation
- **PlaceNet** for candidate place pose generation

## Installation

### Option 1: Install with uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/dgrachev/lerobot_policy_a2
cd lerobot_policy_a2

# Install with uv
uv sync

# Activate the environment
source .venv/bin/activate
```

### Option 2: Install with pip

```bash
# Clone the repository
git clone https://github.com/dgrachev/lerobot_policy_a2
cd lerobot_policy_a2

# Install in development mode
pip install -e .
```

## Quick Start

### Load Pretrained Model from HuggingFace

```python
from lerobot_policy_a2 import A2Policy

# Load pretrained model from HuggingFace
policy = A2Policy.from_pretrained("dgrachev/a2_pretrained")

# Use for grasp prediction
action, info = policy.predict_grasp(
    color_images={"front": rgb_image},
    depth_images={"front": depth_image},
    point_cloud=point_cloud,  # (N, 3) or (N, 6) with colors
    lang_goal="grasp a round object"
)
# action: 7D pose [x, y, z, qx, qy, qz, qw]
```

### Standalone Usage (without LeRobot)

```python
import torch
from lerobot_policy_a2 import A2Config, A2Policy, clip

# Create configuration
config = A2Config(
    width=768,
    layers=1,
    heads=8,
    use_rope=True,
    device="cuda",
)

# Create policy
policy = A2Policy(config)

# Load CLIP model
clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

# Use policy for inference
# pts_pos: Point cloud positions (B, N, 3)
# pts_feat: Point features from CLIP (B, N, D)
# pts_sim: Similarity scores (B, N, 1)
# actions: Candidate action poses (B, M, 7) - xyz + quaternion
logits, selected_action = policy.select_action(pts_pos, pts_feat, pts_sim, actions)
```

## Using with LeRobot A2 Environment

The A2 simulation environment is available in [lerobot_grach0v](https://github.com/grach0v/lerobot) and provides:
- **Robot**: UR5e with Robotiq 2F-85 gripper
- **Tasks**: `grasp`, `place`, `pick_and_place`
- **Object Sets**: `train` (66 simplified objects), `test` (17 unseen objects)
- **Cameras**: `front`, `overview`, `gripper`

### Install LeRobot with A2 Environment

```bash
# Clone LeRobot fork with A2 support
git clone https://github.com/grach0v/lerobot.git lerobot_grach0v
cd lerobot_grach0v

# Install with uv (recommended)
uv sync --extra a2

# Or with pip
pip install -e ".[a2]"
```

### Data Collection with A2 Policy

```bash
cd lerobot_grach0v

# Collect data with pretrained A2 policy from HuggingFace
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_collect \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --task grasp \
    --num_episodes 100 \
    --max_steps 8 \
    --num_objects 8 \
    --output_dir data/a2_collected \
    --repo_id local/a2_grasp \
    --fps 30 \
    --device cuda

# Collect with random policy (for testing)
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_collect \
    --policy random \
    --task grasp \
    --num_episodes 10 \
    --output_dir /tmp/a2_test
```

### Benchmark Evaluation

```bash
cd lerobot_grach0v

# Evaluate A2 policy on grasp task (seen objects)
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --object_set train \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --num_episodes 15 \
    --max_steps 8 \
    --results_dir benchmark_results \
    --device cuda

# Evaluate on unseen objects
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --object_set test \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained \
    --num_episodes 15 \
    --results_dir benchmark_results

# Evaluate specific test case
A2_DISABLE_EGL=true uv run python -m lerobot.envs.a2_benchmark \
    --task grasp \
    --testing_case case00-round.txt \
    --policy a2 \
    --hf_repo dgrachev/a2_pretrained
```

## HuggingFace Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **Pretrained Model** | [dgrachev/a2_pretrained](https://huggingface.co/dgrachev/a2_pretrained) | Trained A2 policy weights + GraspNet checkpoint |
| **Environment Assets** | [dgrachev/a2_assets](https://huggingface.co/datasets/dgrachev/a2_assets) | 3D object models, robot URDFs, testing cases |

## Architecture

### A2Config

```python
@dataclass
class A2Config:
    # Network architecture
    width: int = 768          # Hidden dimension
    layers: int = 1           # Number of cross-attention layers
    heads: int = 8            # Number of attention heads
    action_dim: int = 7       # xyz (3) + quaternion (4)

    # Position encoding
    use_rope: bool = True     # Use Rotary Position Encoding
    no_feat_rope: bool = False

    # Point cloud processing
    sample_num: int = 500     # Number of point cloud samples

    # Feature settings
    feat_backbone: str = "clip"
    fusion_sa: bool = False   # Self-attention after cross-attention
    no_rgb_feat: bool = False # Skip RGB features

    # GraspNet checkpoint (optional)
    graspnet_checkpoint: str = None
```

### A2Policy Pipeline

1. **Depth → Point Cloud**: Extract 3D points from depth image using camera intrinsics
2. **RGB → CLIP Features**: Extract visual features from RGB using CLIP encoder
3. **GraspNet/PlaceNet → Candidates**: Generate candidate 6-DOF poses
4. **Cross-Attention Fusion**: Fuse point features with candidate action embeddings
5. **Policy Head**: Score candidates and select best action

### Network Components

- `CLIPActionFusion`: Cross-attention fusion of point cloud and action features
- `Policy`: MLP head for action scoring
- `RotaryPositionEncoding3D`: 3D RoPE for point cloud positions
- `GraspGenerator`: GraspNet-based grasp pose generation
- `PlaceGenerator`: Place pose generation for pick-and-place
- `CLIPGrasp/CLIPPlace`: Action selection agents

## Package Structure

```
lerobot_policy_a2/
├── src/lerobot_policy_a2/
│   ├── __init__.py
│   ├── configuration_a2.py      # A2Config dataclass
│   ├── modeling_a2.py           # A2Policy model
│   ├── clip/                    # CLIP module
│   │   ├── __init__.py
│   │   ├── clip.py              # CLIP loading utilities
│   │   ├── model.py             # CLIP model architecture
│   │   ├── simple_tokenizer.py  # BPE tokenizer
│   │   ├── interpolate.py       # Positional embedding utils
│   │   └── bpe_simple_vocab_16e6.txt.gz
│   ├── graspnet/                # GraspNet module
│   │   ├── __init__.py
│   │   ├── wrapper.py           # GraspNet wrapper
│   │   └── checkpoint-rs.tar    # Pretrained GraspNet (uploaded to HF)
│   ├── networks/                # Neural network components
│   │   ├── __init__.py
│   │   ├── clip_action.py       # CLIPActionFusion, RoPE, Policy
│   │   ├── action_selection.py  # CLIPGrasp, CLIPPlace
│   │   ├── grasp_generator.py   # GraspNet wrapper
│   │   └── place_generator.py   # Place pose generator
│   └── scripts/
│       └── record_dataset.py    # Dataset recording script
├── pyproject.toml
└── README.md
```

## Environment Details

### Observation Space

| Key | Shape | Description |
|-----|-------|-------------|
| `pixels.image` | (480, 640, 3) | Front camera RGB |
| `pixels.image2` | (480, 640, 3) | Overview camera RGB |
| `pixels.image3` | (480, 640, 3) | Gripper camera RGB |
| `depth` | (480, 640) | Front camera depth |
| `robot_state.joints` | (6,) | Joint angles |
| `robot_state.ee_pos` | (3,) | End-effector position |
| `robot_state.ee_quat` | (4,) | End-effector orientation |
| `robot_state.gripper_angle` | (1,) | Gripper opening |

### Action Space

| Mode | Shape | Description |
|------|-------|-------------|
| `pose` | (7,) | xyz position + quaternion orientation |
| `delta_ee` | (7,) | Delta xyz + delta rotation + gripper |
| `joint` | (7,) | Joint targets (6) + gripper |

## Troubleshooting

### GraspNet Dependencies

If you need full GraspNet functionality with `graspnetAPI`:

```bash
# Pre-install scikit-learn first
pip install scikit-learn

# Then install graspnetAPI with deprecated sklearn package
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install graspnetAPI
```

### EGL Rendering Issues

If you encounter EGL/OpenGL errors in headless environments:

```bash
# Disable EGL rendering
export A2_DISABLE_EGL=true
```

## Citation

If you use this policy in your research, please cite:

```bibtex
@misc{a2_policy,
  author = {Denis Grachev},
  title = {A2 Policy: CLIP-based 6-DOF Grasp and Place Policy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/dgrachev/lerobot_policy_a2}
}
```

## License

Apache-2.0
