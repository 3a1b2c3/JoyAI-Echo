"""
LTX-2 Pipelines: High-level video generation pipelines and utilities.
This package provides ready-to-use pipelines for video generation:
- TI2VidOneStagePipeline: Text/image-to-video in a single stage
- CausalTI2VidPipeline: Causal text/image-to-video rollout
- TI2VidTwoStagesPipeline: Two-stage generation with upsampling
- DistilledPipeline: Fast distilled two-stage generation
- ICLoraPipeline: Image/video conditioning with distilled LoRA
- KeyframeInterpolationPipeline: Keyframe-based video interpolation
- RetakePipeline: Regenerate a time region (retake) of an existing video
- ModelLedger: Central coordinator for loading and building models
For more detailed components and utilities, import from specific submodules
like `ltx_pipelines.utils.media_io` or `ltx_pipelines.utils.constants`.
"""

from typing import TYPE_CHECKING

from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines.retake import RetakePipeline
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

if TYPE_CHECKING:
    from ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline


def __getattr__(name: str):
    """Load the optional causal pipeline only when it is requested."""
    if name == "CausalTI2VidPipeline":
        from ltx_pipelines.causal_ti2vid import CausalTI2VidPipeline

        return CausalTI2VidPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "A2VidPipelineTwoStage",
    "CausalTI2VidPipeline",
    "DistilledPipeline",
    "ICLoraPipeline",
    "KeyframeInterpolationPipeline",
    "RetakePipeline",
    "TI2VidOneStagePipeline",
    "TI2VidTwoStagesPipeline",
]
