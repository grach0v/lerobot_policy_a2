"""Pre- and post-processing pipelines for A2 Policy."""

from typing import Any

import torch

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_a2 import A2Config


def make_a2_pre_post_processors(
    config: A2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the A2 policy.

    The pre-processing pipeline handles normalization, batching, and device placement for model inputs.
    The post-processing pipeline handles unnormalization and moves model outputs back to CPU.

    Args:
        config: The A2 policy configuration object.
        dataset_stats: Dictionary containing dataset statistics (mean, std) for normalization.

    Returns:
        Tuple of (pre-processor pipeline, post-processor pipeline).
    """
    # Input preprocessing steps
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]

    # Add normalization if stats are available and normalization is enabled
    if dataset_stats is not None and config.normalization_mapping:
        input_steps.append(
            NormalizerProcessorStep(
                features={**config.input_features, **config.output_features},
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
                device=config.device,
            )
        )

    # Output postprocessing steps
    output_steps = [
        DeviceProcessorStep(device="cpu"),
    ]

    # Add unnormalization if stats are available
    if dataset_stats is not None and config.normalization_mapping:
        output_steps.insert(
            0,
            UnnormalizerProcessorStep(
                features=config.output_features,
                norm_map=config.normalization_mapping,
                stats=dataset_stats,
            ),
        )

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
