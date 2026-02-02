"""CLIP module for A2 Policy.

Modified from https://github.com/openai/CLIP
"""

from .clip import available_models, load, tokenize

__all__ = ["available_models", "load", "tokenize"]
