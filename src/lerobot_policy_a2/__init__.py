"""A2 Policy package for LeRobot.

This package provides the A2 (Action Prior Alignment) policy for 6-DOF grasp and place
pose selection using CLIP-based cross-attention action selection.

Can be used standalone or as a LeRobot policy plugin.
"""

# Ensure lerobot is installed for plugin functionality
try:
    import lerobot  # noqa: F401
except ImportError as err:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package. "
        "Install with: pip install lerobot"
    ) from err

# Also expose submodules for advanced usage
from . import clip, graspnet, networks
from .configuration_a2 import A2Config
from .modeling_a2 import A2Policy
from .processor_a2 import make_a2_pre_post_processors

__all__ = [
    "A2Config",
    "A2Policy",
    "make_a2_pre_post_processors",
    "clip",
    "networks",
    "graspnet",
]

__version__ = "0.1.0"
