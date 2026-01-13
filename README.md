# LeRobot Policy A2

A2 Policy package for [LeRobot](https://github.com/huggingface/lerobot) - ViLGP3D grasp and place policy using CLIP-based action selection.

## Overview

This package implements the ViLGP3D policy for 6-DOF grasp and place pose selection in tabletop manipulation tasks. It uses:

- **CLIP** for visual feature extraction
- **Cross-attention transformer** for fusing point cloud and action features
- **Rotary Position Encoding (RoPE)** for 3D positional information
- **GraspNet/PlaceNet** for candidate pose generation (bundled)

## Installation

```bash
# Install LeRobot with A2 environment support
pip install lerobot[a2]

# Install this policy package
pip install lerobot_policy_a2

# Or install from source
git clone https://github.com/dgrachev/lerobot_policy_a2
cd lerobot_policy_a2
pip install -e .
```

## Quick Start

### Load Pretrained Model

```python
from lerobot_policy_a2 import A2Policy, A2Config

# Load pretrained model from HuggingFace
policy = A2Policy.from_pretrained("dgrachev/a2_pretrained")
```

### Train with LeRobot

```bash
# Train on grasp task
lerobot-train \
    --policy.type a2 \
    --env.type a2 \
    --env.task grasp \
    --steps 100000

# Train on pick-and-place task
lerobot-train \
    --policy.type a2 \
    --env.type a2 \
    --env.task pick_and_place \
    --steps 200000
```

### Record Dataset

```bash
# Record grasp demonstrations
python -m lerobot_policy_a2.scripts.record_dataset \
    --task grasp \
    --num_episodes 1000 \
    --output data/a2_grasp_1000

# Record pick-and-place demonstrations
python -m lerobot_policy_a2.scripts.record_dataset \
    --task pick_and_place \
    --num_episodes 1000 \
    --output data/a2_pp_1000

# Upload to HuggingFace
python -m lerobot_policy_a2.scripts.record_dataset \
    --task grasp \
    --num_episodes 1000 \
    --output data/a2_grasp \
    --upload dgrachev/a2_grasp_1000
```

## Architecture

### A2Config

```python
@dataclass
class A2Config(PreTrainedConfig):
    # Network architecture
    width: int = 768          # Hidden dimension
    layers: int = 1           # Number of cross-attention layers
    heads: int = 8            # Number of attention heads
    action_dim: int = 7       # xyz + quaternion

    # Position encoding
    use_rope: bool = True     # Use Rotary Position Encoding

    # Point cloud
    sample_num: int = 500     # Number of point cloud samples
```

### A2Policy

The policy follows this pipeline:

1. **Depth → Point Cloud**: Extract 3D points from depth image
2. **RGB → CLIP Features**: Extract visual features from RGB
3. **GraspNet/PlaceNet → Candidates**: Generate candidate poses
4. **Cross-Attention**: Fuse point features with candidate actions
5. **Policy Head**: Score and select best action

## Environment

The A2 environment provides:

- **Task Types**: `grasp`, `pick_and_place`
- **Object Sets**: `train` (66 objects), `test` (17 unseen objects)
- **Cameras**: `front`, `overview`, `gripper`
- **Action Modes**: `pose` (6-DOF), `delta_ee`, `joint`

```bash
# Install environment
pip install lerobot[a2]
```

## Related Resources

- **Environment Assets**: [dgrachev/a2_assets](https://huggingface.co/datasets/dgrachev/a2_assets)
- **Pretrained Model**: [dgrachev/a2_pretrained](https://huggingface.co/dgrachev/a2_pretrained)
- **LeRobot**: [huggingface/lerobot](https://github.com/huggingface/lerobot)

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
