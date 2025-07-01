#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:51:18 2025

@author: shayneufeld
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
import subprocess
from datetime import datetime
from pathlib import Path
import torchaudio
import torch
import numpy as np
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline, Inference
from pyannote.core import Segment
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ------------ Constants ------------
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN") or "<your_huggingface_token_here>"
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
if not HF_AUTH_TOKEN:
    raise ValueError("Missing Hugging Face token. Set HF_AUTH_TOKEN env variable.")
else:
    print(f"[INFO] using HF token from env: {HF_AUTH_TOKEN}")


def convert(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert(v) for v in obj]
    return obj

def extract_audio_chunks(input_path, output_dir, chunk_duration, overlap):
    import ffmpeg
    probe = ffmpeg.probe(input_path)
    duration = float(probe['format']['duration'])

    os.makedirs(output_dir, exist_ok=True)
    chunks = []
    i = 0
    start = 0

    while start < duration:
        chunk_path = os.path.join(output_dir, f"chunk_{i}.wav")
        actual_start = max(0, start - (overlap if i > 0 else 0))
        cmd = (
            ffmpeg
            .input(input_path, ss=actual_start, t=chunk_duration)
            .output(chunk_path, ac=1, ar=16000, format='wav')
            .overwrite_output()
        )
        cmd.run(quiet=True)
        chunks.append({
            "path": chunk_path,
            "start": actual_start,
            "end": min(actual_start + chunk_duration, duration),
            "index": i
        })
        start += chunk_duration
        i += 1

    return chunks


def transcribe_audio(model, audio_path):
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    return {
        "language": info.language,
        "duration": info.duration,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text.strip(),
                "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                          for w in s.words] if s.words else []
            }
            for s in segments
        ]
    }


def perform_diarization(pipeline, embedding_model, audio_path, speaker_count):
    diarization = pipeline(audio_path, num_speakers=speaker_count if speaker_count > 0 else None)
    result = {"segments": [], "embeddings": {}}

    waveform, sample_rate = torchaudio.load(audio_path)

    # Make sure it's mono (1 channel) and sample rate is 16kHz
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    file_input = {"waveform": waveform, "sample_rate": sample_rate}
            
    with torch.no_grad():
        waveform_duration = waveform.shape[1] / sample_rate
    
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = Segment(turn.start, turn.end)
    
            # Clip segment to avoid crop-out-of-bounds error
            if segment.end > waveform_duration:
                segment = Segment(segment.start, min(segment.end, waveform_duration))
    
            embedding = embedding_model.crop(file_input, segment)
            vec = embedding.data.mean(axis=0).tolist()
    
            result["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
                "embedding": vec
            })
            result["embeddings"].setdefault(speaker, []).append(vec)

    for speaker in result["embeddings"]:
        result["embeddings"][speaker] = np.mean(result["embeddings"][speaker], axis=0).tolist()

    return result


def align_transcription_and_diarization(transcription, diarization):
    aligned = []
    for ts in transcription["segments"]:
        matching = next((spk for spk in diarization["segments"] if spk["start"] <= ts["start"] <= spk["end"]), None)
        aligned.append({
            **ts,
            "speaker": matching["speaker"] if matching else "SPEAKER_UNKNOWN",
            "embedding": matching.get("embedding") if matching else None
        })
    return aligned


def cluster_speakers(segments, speaker_count):
    embeddings = [s["embedding"] for s in segments if s.get("embedding")]
    if not embeddings:
        for s in segments:
            s["global_speaker"] = 0
        return segments

    X = np.array(embeddings)
    X = StandardScaler().fit_transform(X)

    if speaker_count > 0:
        clusterer = KMeans(n_clusters=speaker_count, n_init=10, random_state=42)
    else:
        sim = cosine_similarity(X)
        eps = 1 - np.median(sim[sim < 0.99])
        clusterer = DBSCAN(eps=eps, min_samples=2, metric='cosine')

    labels = clusterer.fit_predict(X)
    label_map = {}
    counter = 0

    for i, s in enumerate(segments):
        if s.get("embedding"):
            label = labels[counter]
            counter += 1
        else:
            label = -1
        s["global_speaker"] = label if label != -1 else 0

    return segments


def merge_segments(segments):
    segments.sort(key=lambda s: s["start"])
    merged = []
    last_end = 0
    for s in segments:
        if s["start"] >= last_end:
            merged.append(s)
            last_end = s["end"]
    return merged


def write_outputs(segments, output_path, model_name, speaker_count):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(f"{output_path}.txt", "w") as f:
        f.write(f"# Meeting Transcript\n\nGenerated: {datetime.now().isoformat()}\n")
        f.write(f"Model: {model_name} | Speakers: {speaker_count}\n\n---\n\n")
        for s in segments:
            mins, secs = divmod(int(s["start"]), 60)
            f.write(f"**Speaker {s['global_speaker'] + 1}** [{mins}:{secs:02d}]: {s['text']}\n\n")

    with open(f"{output_path}.json", "w") as f:
        json.dump({"segments": convert(segments)}, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Meeting Transcriber CLI")
    parser.add_argument("input_file", help="Path to input audio file (e.g. .m4a)")
    parser.add_argument("-m", "--model", default="base", help="Whisper model (e.g. base, small)")
    parser.add_argument("-s", "--speakers", type=int, default=0, help="Number of speakers (0 = auto)")
    parser.add_argument("-o", "--output", default="./transcription_output", help="Output directory")
    parser.add_argument("--chunk", type=int, default=300, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=int, default=30, help="Overlap between chunks in seconds")
    args = parser.parse_args()

    input_path = args.input_file
    output_dir = args.output
    base_name = Path(input_path).stem
    output_path = os.path.join(output_dir, base_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("[INFO] Extracting audio chunks...")
        chunks = extract_audio_chunks(input_path, tmpdir, args.chunk, args.overlap)

        print(f"[INFO] Loading Whisper model: {args.model}")
        whisper_model = WhisperModel(args.model, device="auto", compute_type="int8")

        print("[INFO] Loading diarization models")
        
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_AUTH_TOKEN)
        embedding_model = Inference("pyannote/embedding", use_auth_token=HF_AUTH_TOKEN)

        all_segments = []

        for chunk in chunks:
            print(f"[INFO] Processing chunk {chunk['index'] + 1}/{len(chunks)}...")
            transcription = transcribe_audio(whisper_model, chunk["path"])
            diarization = perform_diarization(pipeline, embedding_model, chunk["path"], args.speakers)
            aligned = align_transcription_and_diarization(transcription, diarization)

            for seg in aligned:
                seg["start"] += chunk["start"]
                seg["end"] += chunk["start"]
                all_segments.append(seg)

        print("[INFO] Clustering speakers...")
        clustered = cluster_speakers(all_segments, args.speakers)
        final = merge_segments(clustered)

        print(f"[INFO] Writing output to: {output_path}")
        write_outputs(final, output_path, args.model, args.speakers)

        print("[✅] Transcription complete!")


if __name__ == "__main__":
    main()
