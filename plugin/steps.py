"""Custom steps for video-to-image upload plugin."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import cv2

from synapse_sdk.plugins.actions.upload.context import UploadContext
from synapse_sdk.plugins.steps import BaseStep, StepResult


class ExtractVideoFramesStep(BaseStep[UploadContext]):
    """Extract frames from video files and replace organized_files with frame entries.

    Reads extra_params from context:
        - extracted_frame_per_second (float | None): FPS to extract. None = all frames.
        - output_format (str): Output image format ('png' or 'jpg'). Default: 'png'.
        - group_name (str | None): Group name to assign to all data units.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    @property
    def name(self) -> str:
        return 'extract_video_frames'

    @property
    def progress_weight(self) -> float:
        return 0.15

    def can_skip(self, context: UploadContext) -> bool:
        """Skip if no video files found in organized_files."""
        for file_group in context.organized_files:
            for file_path in file_group.get('files', {}).values():
                if isinstance(file_path, list):
                    file_path = file_path[0] if file_path else None
                if file_path and Path(file_path).suffix.lower() in self.VIDEO_EXTENSIONS:
                    return False
        return True

    def execute(self, context: UploadContext) -> StepResult:
        extra = context.params.get('extra_params') or {}
        extracted_fps = self._parse_fps(extra.get('extracted_frame_per_second'))
        output_format = extra.get('output_format', 'png')
        group_name = extra.get('group_name')

        temp_dir = self._create_temp_directory(context)
        processed_files: list[dict[str, Any]] = []
        total_frames_extracted = 0

        try:
            for file_group in context.organized_files:
                files_dict = file_group.get('files', {})
                meta = file_group.get('meta', {})

                for spec_name, file_path in files_dict.items():
                    if isinstance(file_path, list):
                        file_path = file_path[0] if file_path else None
                    if file_path is None:
                        continue

                    # Preserve original object for SFTP handling, convert to Path only for local files
                    original_file_path = file_path
                    path_for_check = Path(file_path) if isinstance(file_path, str) else file_path

                    # Get suffix for extension check
                    if hasattr(path_for_check, 'suffix'):
                        suffix = path_for_check.suffix.lower()
                    elif hasattr(path_for_check, 'name'):
                        suffix = Path(path_for_check.name).suffix.lower()
                    else:
                        suffix = Path(str(file_path)).suffix.lower()

                    if suffix not in self.VIDEO_EXTENSIONS:
                        processed_files.append(file_group)
                        continue

                    # Use original_file_path for SFTP support, path_for_check for metadata
                    extracted_frames, video_metadata = self._extract_frames(
                        original_file_path, temp_dir, extracted_fps, output_format, context,
                    )

                    # Get file name for logging
                    if hasattr(path_for_check, 'name'):
                        file_name = path_for_check.name
                    else:
                        file_name = Path(str(original_file_path)).name

                    if not extracted_frames:
                        context.log(
                            'video_frame_extraction_skip',
                            {'file': file_name, 'reason': 'no frames extracted'},
                        )
                        continue

                    for i, frame_path in enumerate(extracted_frames):
                        frame_meta = {
                            **meta,
                            'origin_file_name': file_name,
                            'origin_file_format': suffix.lstrip('.'),
                            'fps': video_metadata.get('fps', 0),
                            'resolution': video_metadata.get('resolution', ''),
                            'width': video_metadata.get('width', 0),
                            'height': video_metadata.get('height', 0),
                            'total_frames': video_metadata.get('total_frames', 0),
                            'duration': video_metadata.get('duration', 0),
                            'fourcc': video_metadata.get('fourcc_str', ''),
                            'frame_count': len(extracted_frames),
                            'frame_index': i + 1,
                            'extracted_fps': extracted_fps if extracted_fps else 'all',
                            'output_format': output_format,
                        }

                        entry: dict[str, Any] = {
                            'files': {spec_name: Path(frame_path)},
                            'meta': frame_meta,
                        }
                        if group_name:
                            entry['groups'] = [group_name]

                        processed_files.append(entry)

                    total_frames_extracted += len(extracted_frames)

            context.organized_files = processed_files

            # Register temp directory for cleanup after upload completes
            context.params['cleanup_temp'] = True
            context.params['temp_path'] = str(temp_dir)

            context.log(
                'video_frame_extraction_complete',
                {'total_frames': total_frames_extracted, 'total_entries': len(processed_files)},
            )

            return StepResult(
                success=True,
                data={'frames_extracted': total_frames_extracted},
                rollback_data={'temp_dir': str(temp_dir)},
            )

        except Exception as e:
            return StepResult(success=False, error=f'Video frame extraction failed: {e}')

    def rollback(self, context: UploadContext, result: StepResult) -> None:
        temp_dir = result.rollback_data.get('temp_dir')
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_fps(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            fps = float(value)
            return fps if fps > 0 else None
        except (ValueError, TypeError):
            return None

    def _create_temp_directory(self, context: UploadContext) -> Path:
        base = context.pathlib_cwd if context.pathlib_cwd else Path(os.getcwd())
        temp_dir = base / 'temp_video_frames'
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def _get_video_metadata(self, cap: cv2.VideoCapture) -> dict[str, Any]:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        fourcc_str = fourcc.to_bytes(4, byteorder='little').decode('ascii', errors='ignore')
        duration = total_frames / fps if fps > 0 else 0

        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'fourcc_str': fourcc_str,
            'duration': duration,
            'resolution': f'{width}x{height}',
        }

    def _resolve_video_path(self, input_path: Any, context: UploadContext) -> tuple[str, Path | None]:
        """Resolve video path, downloading from SFTP if necessary.

        Supports both local Path objects and SFTP/remote path objects.

        Returns:
            (local video file path string, temp_file Path to clean up or None)
        """
        # Check for SFTP or remote path object (has open/name but is not a local Path)
        is_remote = (
            hasattr(input_path, 'open')
            and hasattr(input_path, 'name')
            and not isinstance(input_path, Path)
            and not isinstance(input_path, str)
        )

        if is_remote:
            # SFTP or remote path - download to local temp file
            temp_dir = self._create_temp_directory(context) / 'videos'
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / f'temp_video_{input_path.name}'

            with input_path.open('rb') as src, open(temp_file, 'wb') as dst:
                dst.write(src.read())

            return str(temp_file), temp_file

        if hasattr(input_path, 'exists') and not input_path.exists():
            context.log(
                'video_file_not_found', {'file': str(input_path)},
            )
            return '', None

        return str(input_path), None

    def _extract_frames(
        self,
        video_path: Any,
        output_dir: Path,
        extracted_fps: float | None,
        output_format: str,
        context: UploadContext,
    ) -> tuple[list[str], dict[str, Any]]:
        """Extract frames from a single video file.

        Supports both local Path objects and SFTP/remote path objects.

        Returns:
            (list of extracted frame paths, video metadata dict)
        """
        video_file_path, temp_file = self._resolve_video_path(video_path, context)
        if not video_file_path:
            return [], {}

        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            return [], {}

        try:
            metadata = self._get_video_metadata(cap)
            video_fps = metadata['fps']
            total_frames = metadata['total_frames']

            if extracted_fps is not None and extracted_fps > 0:
                frame_interval = video_fps / extracted_fps
            else:
                frame_interval = 1

            ext = output_format if output_format.startswith('.') else f'.{output_format}'

            # Get stem from video_path (supports both Path and SFTP objects)
            if hasattr(video_path, 'stem'):
                stem = video_path.stem
            elif hasattr(video_path, 'name'):
                stem = Path(video_path.name).stem
            else:
                stem = Path(str(video_path)).stem

            # Get name for logging (supports both Path and SFTP objects)
            if hasattr(video_path, 'name'):
                video_name = video_path.name
            else:
                video_name = Path(str(video_path)).name

            extracted_files: list[str] = []
            current_frame = 0
            saved_frames = 0
            next_frame_to_extract = 0.0

            while current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame >= next_frame_to_extract:
                    frame_filename = f'{stem}_{current_frame:06d}{ext}'
                    frame_path = output_dir / frame_filename

                    if cv2.imwrite(str(frame_path), frame):
                        saved_frames += 1
                        extracted_files.append(str(frame_path))
                        next_frame_to_extract += frame_interval

                        if saved_frames % 100 == 0:
                            progress = (current_frame / total_frames) * 100
                            context.log(
                                'video_frame_extraction_progress',
                                {'file': video_name, 'frames': saved_frames, 'progress': f'{progress:.1f}%'},
                            )

                current_frame += 1

            context.log(
                'video_frames_extracted',
                {'file': video_name, 'total_frames': saved_frames},
            )
            return extracted_files, metadata

        finally:
            cap.release()
            if temp_file and temp_file.exists():
                temp_file.unlink()
