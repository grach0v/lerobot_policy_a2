# lerobot_policy_a2 - Migration Status

## What this package is
Standalone policy package for A2 (ViLGP3D). Can be pip installed and used with lerobot or independently.

---

## Current Status: MOSTLY COMPLETE

### ✅ Already done:
- GraspNet model (`graspnet/`)
  - Full model code from A2_new/models/graspnet_new/
  - Pure PyTorch fallbacks for pointnet2 CUDA ops (`pointnet2_ops_fallback.py`)
  - Pure PyTorch fallback for KNN (`knn_fallback.py`)
  - `get_graspnet_checkpoint()` for HuggingFace download

- PlaceNet (`networks/place_generator.py`)
  - Place pose generation

- CLIP module (`clip/`)
  - Visual encoder for feature extraction

- Action selection networks (`networks/`)
  - `action_selection.py`
  - `clip_action.py`
  - `grasp_generator.py` - uses local GraspNet

- Configuration (`configuration_a2.py`)
  - A2Config with LeRobot PreTrainedConfig

- Policy (`modeling_a2.py`)
  - A2Policy implementation

### ✅ Import test passes:
```python
from lerobot_policy_a2 import graspnet
# GraspNet module imported successfully
# Available: ['GraspNetBaseLine', 'vis_grasps', 'get_graspnet_checkpoint']
```

---

## TODO (Minor):

### 1. Upload GraspNet checkpoint to HuggingFace
- Current checkpoint: `A2_new/models/graspnet_new/checkpoint-rs.tar`
- Upload to: `dgrachev/graspnet_checkpoint`
- The download function already exists in `graspnet/wrapper.py`

### 2. Verify policy weights can be loaded
- Test `A2Policy.from_pretrained()` works
- May need to upload trained weights to HuggingFace

### 3. Remove pybullet dependency (optional)
- Currently in pyproject.toml but policy doesn't need simulation
- Only needed if policy is tested standalone with env

---

## Package structure:
```
src/lerobot_policy_a2/
├── __init__.py
├── configuration_a2.py      # A2Config
├── modeling_a2.py           # A2Policy
├── clip/                    # CLIP visual encoder
│   ├── __init__.py
│   ├── clip.py
│   ├── model.py
│   └── simple_tokenizer.py
├── graspnet/                # GraspNet for grasp detection
│   ├── __init__.py
│   ├── wrapper.py           # GraspNetBaseLine, get_graspnet_checkpoint
│   ├── models/
│   │   ├── graspnet.py
│   │   ├── backbone.py
│   │   ├── modules.py
│   │   └── loss.py
│   ├── pointnet2/
│   │   ├── pointnet2_utils.py
│   │   ├── pointnet2_modules.py
│   │   ├── pointnet2_ops_fallback.py  # Pure PyTorch fallback
│   │   └── pytorch_utils.py
│   ├── knn/
│   │   ├── knn_modules.py
│   │   └── knn_fallback.py   # Pure PyTorch fallback
│   └── utils/
│       ├── collision_detector.py
│       ├── data_utils.py
│       ├── label_generation.py
│       └── loss_utils.py
├── networks/
│   ├── __init__.py
│   ├── action_selection.py
│   ├── clip_action.py
│   ├── grasp_generator.py   # Uses local graspnet
│   └── place_generator.py   # PlaceNet
└── scripts/
    └── (empty for now)
```

---

## Usage after install:
```bash
pip install lerobot_policy_a2

# In Python:
from lerobot_policy_a2 import A2Policy, A2Config
from lerobot_policy_a2.graspnet import GraspNetBaseLine, get_graspnet_checkpoint

# Load policy
policy = A2Policy.from_pretrained("dgrachev/a2_policy")

# Or use GraspNet directly
checkpoint = get_graspnet_checkpoint()  # Downloads from HuggingFace
graspnet = GraspNetBaseLine(checkpoint)
grasps = graspnet.inference(point_cloud)
```

---

## Dependencies (pyproject.toml):
- torch, torchvision
- numpy, scipy, scikit-learn
- open3d, open3d_plus
- Pillow
- huggingface_hub
- graspnetAPI
- ftfy, regex, tqdm, packaging

Note: graspnetAPI may have install issues - the pure PyTorch fallbacks work without CUDA extensions.
