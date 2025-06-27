import os
import sys
import json
import tempfile
import argparse
import logging
from datetime import timedelta

import torch
import torchaudio
import ffmpeg
import numpy as np
from tqdm import tqdm
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

# ------------------------
# Utility Functions
# ------------------------
def get_user_inputs():
    model_size = input("Choose whisper model size (tiny, base, small, medium, large): ").strip()
    num_speakers = input("Enter number of speakers (or press Enter to skip): ").strip()
    num_speakers = int(num_speakers) if num_speakers else None
    return model_size, num_speakers

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    return str(td).split(".")[0]

# ------------------------
# Core Pipeline Functions
# ------------------------
def extract_audio(input_path, output_path, force=False):
    if os.path.exists(output_path) and not force:
        logger.info("[1/6] Skipping audio extraction (already exists)")
        return
    logger.info("[1/6] Extracting audio from input file...")
    (
        ffmpeg.input(input_path)
        .output(output_path, format='wav', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(f"    Audio saved to: {output_path}")

def transcribe_audio(audio_path, model_size, output_path, force=False):
    if os.path.exists(output_path) and not force:
        logger.info("[2/6] Skipping transcription (already exists)")
        return output_path
    logger.info("[2/6] Transcribing audio with Faster-Whisper...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    data = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    with open(output_path, "w") as f:
        json.dump({"segments": data}, f, indent=2)
    logger.info(f"    Transcript saved to: {output_path}")
    return output_path

def diarize_audio(audio_path, diar_path, force=False):
    if os.path.exists(diar_path) and not force:
        logger.info("[3/6] Skipping diarization (already exists)")
        return diar_path
    logger.info("[3/6] Running diarization with pyannote.audio...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_AUTH_TOKEN)
    diarization = pipeline(audio_path)
    with open(diar_path, "w") as f:
        diarization.write_rttm(f)
    logger.info(f"    Diarization RTTM saved to: {diar_path}")
    return diar_path

def embed_and_cluster_segments(audio_path, diarization, num_speakers):
    logger.info("[4/6] Embedding and clustering segments with SpeechBrain...")
    recog = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},
    )
    waveform, sr = torchaudio.load(audio_path)
    embeddings = []
    times = []
    for turn, _, _ in tqdm(diarization.itertracks(yield_label=True), desc="Embedding segments"):
        start, end = int(turn.start * sr), int(turn.end * sr)
        segment = waveform[:, start:end]
        try:
            emb = recog.encode_batch(segment).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            times.append((turn.start, turn.end))
        except Exception as e:
            logger.warning(f"Failed to embed segment {turn}: {e}")
    if len(embeddings) < 2:
        raise ValueError("Need at least two valid embeddings for clustering.")
    X = np.vstack(embeddings)
    clusterer = AgglomerativeClustering(n_clusters=num_speakers) if num_speakers else AgglomerativeClustering(n_clusters=None, distance_threshold=0.6)
    labels = clusterer.fit_predict(X)
    return [{"start": s, "end": e, "speaker": f"Speaker {l}"} for (s, e), l in zip(times, labels)]

def merge_transcript(transcript_path, labeled_segments, out_path):
    logger.info("[5/6] Merging transcript with speaker labels...")
    with open(transcript_path) as f:
        data = json.load(f)
    result = []
    for seg in data["segments"]:
        t = (seg["start"] + seg["end"]) / 2
        speaker = next((s["speaker"] for s in labeled_segments if s["start"] <= t <= s["end"]), "Unknown")
        result.append(f"[{format_timestamp(seg['start'])}] {speaker}: {seg['text']}")
    with open(out_path, "w") as f:
        f.write("\n".join(result))
    logger.info(f"    Final transcript written to {out_path}")

# ------------------------
# Main Entrypoint
# ------------------------
def main(input_path, output_dir, model_size, num_speakers, force=False):
    os.makedirs(output_dir, exist_ok=True)
    tmp_wav = os.path.join(tempfile.gettempdir(), "audio.wav")
    transcript_json = os.path.join(output_dir, f"transcript.{model_size}.json")
    diar_rttm = os.path.join(output_dir, "diarization.rttm")
    final_txt = os.path.join(output_dir, "final_transcript.txt")

    extract_audio(input_path, tmp_wav, force=force)
    transcribe_audio(tmp_wav, model_size, transcript_json, force=force)
    diarize_audio(tmp_wav, diar_rttm, force=force)

    from pyannote.core import Annotation
    diarization = Annotation()
    with open(diar_rttm) as f:
        diarization.load_rttm(f)

    labeled = embed_and_cluster_segments(tmp_wav, diarization, num_speakers)
    merge_transcript(transcript_json, labeled, final_txt)
    logger.info("[6/6] Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input MP4/M4A/WAV file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force overwrite of intermediate outputs")
    args = parser.parse_args()
    model_size, num_speakers = get_user_inputs()
    main(args.input_path, args.output_dir, model_size, num_speakers, force=args.force)
