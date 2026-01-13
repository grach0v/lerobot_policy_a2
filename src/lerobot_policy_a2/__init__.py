"""A2 Policy package for LeRobot.

This package provides the ViLGP3D policy for 6-DOF grasp and place pose selection
using CLIP-based cross-attention action selection.

Can be used standalone or as a LeRobot policy plugin.
"""

from .configuration_a2 import A2Config
from .modeling_a2 import A2Policy
from . import clip
from . import networks

__all__ = [
    "A2Config",
    "A2Policy",
    "clip",
    "networks",
]

__version__ = "0.1.0"
