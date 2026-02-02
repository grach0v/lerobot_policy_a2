"""GraspNet for 6-DOF grasp pose generation.

Based on GraspNet-PointNet2-Pytorch-General-Upgrade by H-Freax.
https://github.com/H-Freax/GraspNet-PointNet2-Pytorch-General-Upgrade
"""

import sys
from pathlib import Path

# Add subdirectories to path for relative imports
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "models"))
sys.path.insert(0, str(_ROOT / "pointnet2"))
sys.path.insert(0, str(_ROOT / "utils"))

from .wrapper import GraspNetBaseLine, get_graspnet_checkpoint, vis_grasps

__all__ = ["GraspNetBaseLine", "vis_grasps", "get_graspnet_checkpoint"]
