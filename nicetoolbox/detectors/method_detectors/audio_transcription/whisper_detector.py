"""
Whisper method detector class (mock/debug implementation).

Tests that the audio pipeline — from data preparation through
AudioStreamLoader — works end-to-end. The actual Whisper inference
is replaced with a lightweight validation that logs audio metadata.
"""

import logging

from nicetoolbox_core.audio_loaders import AudioStreamLoader

from ....configs.schemas.detectors_algos_configs import MethodDetectorRuntime
from ..base_method import BaseMethod

logger = logging.getLogger(__name__)


class Whisper(BaseMethod):
    """
    Mock Whisper detector for audio pipeline validation.

    Instead of running real transcription, this:
    1. Constructs an AudioStreamLoader from the input recipes
    2. Iterates all tracks and logs metadata
    3. Verifies file existence and timing alignment

    This is sufficient to validate the full audio pipeline during debugging.
    """

    algorithm = "whisper"
    components = ["audio_transcription"]

    def _initialize_detector(self) -> MethodDetectorRuntime:
        """
        Initialize Whisper detector.

        Validates that audio recipes are available and constructs
        an AudioStreamLoader to verify the pipeline.
        """
        # Verify audio is available
        if not self.data.has_audio():
            raise RuntimeError(
                "Whisper detector requires audio data but no audio was prepared. "
                "Check that input_data_types = ['audio'] is set in detectors_config.toml "
                "and that audio tracks are configured in dataset_properties.toml."
            )

        # Construct audio loader from recipes — this validates the recipe structure
        input_recipes = self.data.get_input_recipes()
        self.audio_loader = AudioStreamLoader(config=input_recipes, expected_tracks=None)

        logger.info(
            f"Whisper: AudioStreamLoader initialized with {len(self.audio_loader)} track(s): "
            f"{self.audio_loader.tracks}"
        )
        logger.info(
            f"Whisper: Time range: offset={self.audio_loader.offset_seconds:.3f}s, "
            f"duration={self.audio_loader.duration_seconds:.3f}s"
        )

        # Log per-track info
        for track_name in self.audio_loader.tracks:
            info = self.audio_loader.get_stream_info(track_name)
            logger.info(
                f"  Track '{track_name}': "
                f"path={info['path']}"
                f"sr={info['sample_rate']}Hz, "
                f"channels={info['channels']}, "
                f"source={info['source_type']}, "
                f"hears={info['hears_subjects']}, "
            )

        # Build base runtime (will be serialized to TOML for subprocess)
        return super()._initialize_detector()

    def post_inference(self) -> None:
        """
        Post-inference hook.

        For the mock detector, we validate the audio loader iteration
        in-process instead of relying on the subprocess.
        """
        logger.info("Whisper (mock): Running in-process audio loader validation...")

        for track_name, path, offset, duration in self.audio_loader:
            logger.info(f"Track '{track_name}': {path} " f"(offset={offset:.3f}s, duration={duration:.3f}s)")

        # Check subject-track mapping
        for i, subject in enumerate(self.data.subjects_descr):
            tracks = self.audio_loader.get_tracks_for_subject(i)
            logger.info(f"Subject '{subject}' (idx={i}) heard by tracks: {tracks}")

        logger.info("Whisper (mock): Audio pipeline validation complete.")

    def visualization(self, data) -> None:
        pass
