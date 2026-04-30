"""
WhisperX inference entrypoint.
"""

import json
import logging
import os

import librosa
import torch
import whisperx

from nicetoolbox_core.audio_loaders import AudioStreamLoader
from nicetoolbox_core.entrypoint import run_inference_entrypoint


# TODO: this is a bit hacky - pyannote's diarization segments contain Segment objects which are not json serializable
class PyannoteEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "start") and hasattr(obj, "end"):  # Pyannote Segment object inside dict...
            return {"start": obj.start, "end": obj.end}
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)


@run_inference_entrypoint
def whisperx_inference(config: dict) -> None:
    """
    Executes WhisperX transcription pipeline natively isolated by the environment.
    """
    logging.info("Starting WhisperX Inference Pipeline.")

    algo_name = config["algorithm"]
    res_folders = config["result_folders"]  # top level json (npz) results
    extra_detector_output_folder = config["out_folder"]  # additional detector results (e.g. SRT files)
    assets_dir = config["required_assets"]["base_dir"]

    model_size = config["model_size"]
    language = config["language"]
    batch_size = config["batch_size"]
    vad_onset = config["vad_onset"]
    vad_offset = config["vad_offset"]
    align_model_name = config["alignment_model_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = config["compute_type"] if device == "cuda" else "int8"

    audio_loader = AudioStreamLoader(config=config, expected_tracks=config["track_names"])

    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=language,
        vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset},
        download_root=assets_dir,
    )
    align_model, metadata = whisperx.load_align_model(
        language_code=language,
        device=device,
        model_name=align_model_name,
        model_dir=assets_dir,
    )
    diarize_model = whisperx.diarize.DiarizationPipeline(token=config["hf_token"], device=device, cache_dir=assets_dir)

    # Components output storage dicts
    out_transcription = {}
    out_diarization = {}

    for track_name, path, offset, duration in audio_loader:
        logging.info(f"Processing track: {track_name}")

        # (1) Load and slice audio track
        sample_rate = 16000  # WhisperX default
        audio, _ = librosa.load(path, sr=sample_rate, offset=offset, duration=duration)
        hears_subjects = audio_loader.get_hears_subjects(track_name)
        logging.info(f"Track '{track_name}' hears subjects: {hears_subjects}")

        # (2) Transcribe
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        out_transcription[track_name] = result

        # (3) Alignment
        result_aligned = whisperx.align(
            result["segments"], align_model, metadata, audio, device, return_char_alignments=False
        )

        # (4) Diarization
        num_speakers = len(hears_subjects)
        diarize_segments = diarize_model(audio, min_speakers=num_speakers, max_speakers=num_speakers)
        out_diarization[track_name] = diarize_segments.to_dict(orient="records")

        # (5) Assign Speakers
        result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)
        result_final["language"] = language

        # (6) Save extra detector output (SRT and JSON files)
        formats = ["srt", "json"]
        os.makedirs(extra_detector_output_folder, exist_ok=True)
        for f in formats:
            writer = whisperx.utils.get_writer(f, extra_detector_output_folder)
            writer_options = {
                "highlight_words": True,
                "max_line_width": None,
                "max_line_count": None,
            }
            writer(result_final, f"{track_name}.wav", writer_options)

    # Save final results for base transcription and diarization
    # NOTE: speaker_aligned_transcription component output is further processed given the individual
    # track json results inside the post_inference function in whisperx_detector.py
    for component, out_dict in [
        ("audio_transcription", out_transcription),
        ("audio_diarization", out_diarization),
    ]:
        comp_dir = res_folders[component]
        os.makedirs(comp_dir, exist_ok=True)
        with open(os.path.join(comp_dir, f"{algo_name}.json"), "w") as f:
            json.dump(out_dict, f, indent=4, cls=PyannoteEncoder)

    logging.info("WhisperX processing complete.")
