# Transcribe Meeting

A local Python script to transcribe and diarize meetings from video/audio files. This tool uses **Faster-Whisper** for fast transcription and **pyannote.audio** for speaker diarization. It outputs clean, timestamped text and a structured JSON file for easy review, summarization, or integration with LLMs.

## Features

- Converts `.mp4`, `.m4a`, or `.wav` meeting recordings to text
- Diarizes speakers using unsupervised clustering
- Supports automatic audio extraction
- Outputs:
  - Clean text transcript
  - Structured `.json` file
- Interactive CLI for choosing:
  - Whisper model size (e.g. `tiny`, `base`, `small`, `medium`, `large`)
  - Language (currently English)
  - Number of speakers (for diarization)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/transcribe-meeting.git
cd transcribe-meeting
```

### 2. Set-up Conda Environment

```bash
# Create a new environment
conda create -n transcribe-meeting python=3.10 -y
conda activate transcribe-meeting
```

# Install required packages
> ```txt
> faster-whisper
> torchaudio
> pyannote.audio
> pydub
> ffmpeg-python
> tqdm
> numpy
> ```

You’ll also need `ffmpeg` installed system-wide. On macOS:

```bash
brew install ffmpeg
```

---

## Usage

```bash
python transcribe-meeting.py /path/to/input.mp4 --output_dir /path/to/output
```

You will be prompted to choose:
- Whisper model (e.g. `tiny`, `base`, `small`, etc.)
- Number of speakers
- Language (English is assumed)

**Basic Example:**
```bash
python transcribe-meeting.py meeting.m4a --output_dir output/
```

### Arguments
| Argument         | Description                                                             |
| ---------------- | ----------------------------------------------------------------------- |
| `input_file`     | (positional) Path to input audio or video file (`.mp4`, `.m4a`, `.wav`) |
| `--output_dir`   | Directory where output files will be saved                              |
| `--model`        | Whisper model to use (`tiny`, `base`, `small`, `medium`, `large`)       |
| `--num_speakers` | Number of speakers to use for diarization (integer > 1)                 |
| `--language`     | Language code for transcription (e.g., `en` for English)                |


**Example with optional arguments:**
```bash
python transcribe-meeting.py recordings/meeting.mp4 \
  --output_dir transcripts/ \
  --model small \
  --num_speakers 2 \
  --language en
```
---

## Outputs

After processing, the following files will appear in the output directory:

- `transcript.txt` – human-readable transcript with speaker labels and timestamps
- `transcript.json` – structured transcript with word-level detail and speaker tags

---

## How It Works

1. **Audio Extraction**: Uses `ffmpeg` to extract audio if input is `.mp4`
2. **Transcription**: Transcribes using `faster-whisper` (fast, quantized Whisper implementation)
3. **Speaker Diarization**: Uses `pyannote.audio` to embed and cluster speaker segments
4. **Output Generation**: Writes `.txt` and `.json` outputs with timestamps and speaker labels

---

## Notes

- Accuracy depends on model size and quality of input audio
- Diarization may require tuning number of speakers
- Whisper model files are downloaded at runtime and cached

---

## License

MIT License. See `LICENSE` file for details.

---

## Acknowledgements

- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [ffmpeg](https://ffmpeg.org/)
