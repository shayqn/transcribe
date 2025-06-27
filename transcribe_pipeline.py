import os
import sys
import tempfile
import json
import time
import torch
import torchaudio
import numpy as np
import ffmpeg
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from speechbrain.inference import SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering
from datetime import timedelta

# ------------ Configurable parameters ------------
HF_AUTH_TOKEN = "hugging-face-token"  # Set your huggingface token here or env var


def get_user_inputs():
    model_size = input("Choose whisper model size (tiny, base, small, medium, large): ").strip()
    num_speakers = input("Enter number of speakers (or press Enter to skip): ").strip()
    num_speakers = int(num_speakers) if num_speakers else None
    return model_size, num_speakers


def extract_audio(input_path, output_wav_path):
    print("[1/6] Extracting audio from video/audio...")
    (
        ffmpeg
        .input(input_path)
        .output(output_wav_path, format='wav', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )
    print(f"    Audio saved to: {output_wav_path}")


def transcribe_audio(audio_path, model_size="small"):
    print("[2/6] Transcribing audio with Faster-Whisper...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    segments_list = []
    for segment in segments:
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    transcript_json_path = audio_path + f".{model_size}.json"
    with open(transcript_json_path, "w") as f:
        json.dump({"segments": segments_list}, f, indent=2)
    print(f"    Transcript JSON saved: {transcript_json_path}")
    return transcript_json_path


def diarize_audio(audio_path, num_speakers=None):
    print("[3/6] Running pyannote.audio diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1",
        use_auth_token=HF_AUTH_TOKEN
    )
    diarization_result = pipeline(audio_path)
    print(f"    Diarization finished, found {len(diarization_result)} speech turns")
    return diarization_result


def embed_and_cluster_segments(diarization_result, audio_path, num_speakers=None):
    print("[4/6] Extracting speaker embeddings and clustering...")

    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="models/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )

    waveform, sample_rate = torchaudio.load(audio_path)

    segment_embeddings = []
    segment_times = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_frame = int(turn.start * sample_rate)
        end_frame = int(turn.end * sample_rate)
        segment_waveform = waveform[:, start_frame:end_frame]

        embedding = speaker_model.encode_batch(segment_waveform).squeeze(0).cpu().numpy()
        segment_embeddings.append(embedding)
        segment_times.append((turn.start, turn.end))

    embeddings_matrix = np.vstack(segment_embeddings)
    print(f"    Extracted {embeddings_matrix.shape[0]} segment embeddings of dim {embeddings_matrix.shape[1]}")

    if num_speakers is None:
        print("    Number of speakers not specified, using distance threshold clustering")
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6).fit(embeddings_matrix)
    else:
        print(f"    Number of speakers specified: {num_speakers}, clustering into fixed clusters")
        clustering = AgglomerativeClustering(n_clusters=num_speakers).fit(embeddings_matrix)

    labels = clustering.labels_
    print(f"    Clustering assigned {len(set(labels))} speakers")

    labeled_segments = []
    for idx, (start, end) in enumerate(segment_times):
        labeled_segments.append({
            "start": start,
            "end": end,
            "speaker": f"Speaker {labels[idx]}"
        })

    return labeled_segments


def get_speaker_for_time(labeled_segments, t):
    for seg in labeled_segments:
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


def merge_transcript_with_speakers(transcript_json_path, labeled_segments, output_txt_path):
    print("[5/6] Merging transcript with speaker labels...")

    with open(transcript_json_path, "r") as f:
        data = json.load(f)
    segments = data.get("segments", [])

    merged = []
    for seg in segments:
        midpoint = (seg["start"] + seg["end"]) / 2
        speaker = get_speaker_for_time(labeled_segments, midpoint)
        merged.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"]
        })

    with open(output_txt_path, "w") as out_file:
        out_file.write("Transcript with speaker labels\n\n")
        for seg in merged:
            ts = format_timestamp(seg["start"])
            out_file.write(f"[{ts}] {seg['speaker']}: {seg['text']}\n")

    print(f"    Transcript with speaker labels saved to {output_txt_path}")


def main(input_path, output_dir, model_size="small", num_speakers=None):
    start_time = time.perf_counter()

    os.makedirs(output_dir, exist_ok=True)

    tmp_wav_path = os.path.join(tempfile.gettempdir(), "audio.wav")

    extract_audio(input_path, tmp_wav_path)

    transcript_json_path = transcribe_audio(tmp_wav_path, model_size=model_size)

    diarization_result = diarize_audio(tmp_wav_path, num_speakers=num_speakers)

    labeled_segments = embed_and_cluster_segments(diarization_result, tmp_wav_path, num_speakers=num_speakers)

    output_txt_path = os.path.join(output_dir, "final_transcript.txt")
    merge_transcript_with_speakers(transcript_json_path, labeled_segments, output_txt_path)

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    print(f"[6/6] All done! Total elapsed time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full transcription + diarization pipeline with consistent speaker labeling")
    parser.add_argument("input_path", help="Path to input audio or video file (mp4/m4a/wav)")
    parser.add_argument("output_dir", help="Directory to save outputs")

    args = parser.parse_args()

    print("Welcome to the transcription pipeline!")
    model_size, num_speakers = get_user_inputs()

    main(args.input_path, args.output_dir, model_size=model_size, num_speakers=num_speakers)
