"""Upload action for video-to-image."""

from __future__ import annotations

from synapse_sdk.plugins.actions.upload import (
    DefaultUploadAction,
    UploadContext,
    UploadParams,
)
from synapse_sdk.plugins.steps import StepRegistry

from plugin.steps import ExtractVideoFramesStep


class UploadAction(DefaultUploadAction[UploadParams]):
    """Upload action that extracts frames from video files before upload.

    Extends the standard 8-step workflow by inserting an ExtractVideoFramesStep
    after organize_files. The custom step converts video files into individual
    image frames (PNG/JPG) and replaces organized_files with frame entries.

    Extra params (via config.yaml ui_schema):
        - extracted_frame_per_second: FPS to extract (empty = all frames)
        - output_format: Output image format (png / jpg)
        - group_name: Group name to assign to all data units
    """

    action_name = 'upload'
    params_model = UploadParams

    def setup_steps(self, registry: StepRegistry[UploadContext]) -> None:
        super().setup_steps(registry)
        registry.insert_after('organize_files', ExtractVideoFramesStep())
