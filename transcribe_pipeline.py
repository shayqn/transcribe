import os
import sys
import tempfile
import json
import torch
import torchaudio
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from datetime import timedelta, datetime
from tqdm import tqdm
import logging

# ------------ Setup Logging ------------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ------------ Constants ------------
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN") or "<your_huggingface_token_here>"

def get_user_inputs():
    model_size = input("Choose whisper model size (tiny, base, small, medium, large): ").strip()
    num_speakers = input("Enter number of speakers (or press Enter to skip): ").strip()
    num_speakers = int(num_speakers) if num_speakers else None
    return model_size, num_speakers

def safe_filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def extract_audio(input_path, output_wav_path, force=False):
    if os.path.exists(output_wav_path) and not force:
        logger.info("[1/6] Skipping audio extraction (already exists)")
        return
    logger.info("[1/6] Extracting audio from video/audio...")
    (
        ffmpeg
        .input(input_path)
        .output(output_wav_path, format='wav', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(f"    Audio saved to: {output_wav_path}")

def transcribe_audio(audio_path, output_json_path, model_size="small", force=False):
    if os.path.exists(output_json_path) and not force:
        logger.info("[2/6] Skipping transcription (already exists)")
        return output_json_path

    logger.info("[2/6] Transcribing audio with Faster-Whisper...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    segments_list = [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
    ]

    with open(output_json_path, "w") as f:
        json.dump({"segments": segments_list}, f, indent=2)

    logger.info(f"    Transcript saved to: {output_json_path}")
    return output_json_path

def diarize_audio(audio_path, diar_path, force=False):
    if os.path.exists(diar_path) and not force:
        logger.info("[3/6] Skipping diarization (already exists)")
        with open(diar_path, "r") as f:
            return json.load(f)

    logger.info("[3/6] Running diarization with pyannote.audio...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_AUTH_TOKEN)
    diarization = pipeline(audio_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    with open(diar_path, "w") as f:
        json.dump(segments, f, indent=2)

    logger.info(f"    Diarization result saved to: {diar_path}")
    return segments

def embed_and_cluster_segments(diarized_segments, audio_path, num_speakers=None):
    logger.info("[4/6] Extracting speaker embeddings and clustering...")

    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    waveform, sr = torchaudio.load(audio_path)

    embeddings = []
    times = []

    for seg in tqdm(diarized_segments, desc="Embedding segments"):
        start, end = int(seg["start"] * sr), int(seg["end"] * sr)
        chunk = waveform[:, start:end]
        emb = model.encode_batch(chunk).squeeze(0).cpu().numpy()
        embeddings.append(emb)
        times.append((seg["start"], seg["end"]))

    matrix = np.vstack(embeddings)

    if num_speakers:
        clusterer = AgglomerativeClustering(n_clusters=num_speakers)
    else:
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6)

    labels = clusterer.fit_predict(matrix)
    logger.info(f"    Found {len(set(labels))} speakers")

    return [
        {"start": start, "end": end, "speaker": f"Speaker {label}"}
        for (start, end), label in zip(times, labels)
    ]

def get_speaker_for_time(segments, t):
    for seg in segments:
        if seg["start"] <= t <= seg["end"]:
            return seg["speaker"]
    return "Unknown"

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def merge_transcript(transcript_json_path, speaker_segments, out_txt_path):
    logger.info("[5/6] Merging transcript with speaker labels...")

    with open(transcript_json_path, "r") as f:
        data = json.load(f)
    segments = data["segments"]

    with open(out_txt_path, "w") as out:
        out.write("Transcript with speaker labels\n\n")
        for seg in segments:
            midpoint = (seg["start"] + seg["end"]) / 2
            speaker = get_speaker_for_time(speaker_segments, midpoint)
            ts = format_timestamp(seg["start"])
            out.write(f"[{ts}] {speaker}: {seg['text']}\n")

    logger.info(f"    Final transcript saved to: {out_txt_path}")

def main(input_path, output_dir, model_size="small", num_speakers=None, force=False):
    os.makedirs(output_dir, exist_ok=True)

    base = safe_filename(input_path)
    time_tag = timestamp()

    tmp_wav = os.path.join(output_dir, f"{base}.{time_tag}.wav")
    json_path = os.path.join(output_dir, f"{base}.{time_tag}.transcript.{model_size}.json")
    diar_path = os.path.join(output_dir, f"{base}.{time_tag}.diarization.json")
    final_txt = os.path.join(output_dir, f"{base}.{time_tag}.final_transcript.txt")

    extract_audio(input_path, tmp_wav, force)
    transcribe_audio(tmp_wav, json_path, model_size=model_size, force=force)
    diar_segments = diarize_audio(tmp_wav, diar_path, force)
    labeled_segments = embed_and_cluster_segments(diar_segments, tmp_wav, num_speakers)
    merge_transcript(json_path, labeled_segments, final_txt)

    logger.info("[6/6] Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument("--force", action="store_true", help="Force rerun of all steps")
    args = parser.parse_args()

    model_size, num_speakers = get_user_inputs()
    main(args.input_path, args.output_dir, model_size=model_size, num_speakers=num_speakers, force=args.force)
