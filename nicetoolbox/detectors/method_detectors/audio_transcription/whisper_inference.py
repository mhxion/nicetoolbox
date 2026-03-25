"""
Mock Whisper inference script for subprocess execution.

This script is launched by BaseMethod._run_inference() in a separate
virtual environment. It validates that the audio recipe survives
TOML serialization and can be consumed by AudioStreamLoader.

For a real implementation, this would:
1. Load audio with librosa using the recipe's timing
2. Run whisper.load_model() and transcribe
3. Save results as npz/json
"""

import logging

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint

logger = logging.getLogger(__name__)


@run_inference_entrypoint
def whisper_inference(config: dict) -> None:
    """
    Mock Whisper inference entrypoint.

    Args:
        config: Flattened config dict parsed from TOML.
            Contains 'input_recipes' with 'audio_input_recipe' nested inside.
    """
    logger.info("Whisper inference (mock): Starting...")

    audio_loader = AudioStreamLoader(config=config)

    logger.info(f"Whisper inference (mock): Loaded {len(audio_loader)} track(s): " f"{audio_loader.tracks}")
    logger.info(
        f"Whisper inference (mock): Time range: "
        f"offset={audio_loader.offset_seconds:.3f}s, "
        f"duration={audio_loader.duration_seconds:.3f}s"
    )

    # Iterate tracks — validates file paths exist
    for track_name, path, offset, duration in audio_loader:
        logger.info(f"    Track '{track_name}': {path}")
        sr = audio_loader.get_sample_rate(track_name)
        logger.info(f"    Sample rate: {sr}Hz, offset: {offset:.3f}s, duration: {duration:.3f}s")

        # In a real implementation:
        # import librosa
        # audio, sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
        # result = model.transcribe(audio)
        # save_results(result, config["result_folders"][...])

    logger.info("Whisper inference (mock): Complete.")
