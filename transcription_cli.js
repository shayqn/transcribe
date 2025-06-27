#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');
const { program } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const ffmpeg = require('fluent-ffmpeg');

class MeetingTranscriber {
  constructor(options = {}) {
    this.modelType = options.model || 'whisper-base';
    this.speakerCount = options.speakers || 2;
    this.chunkDuration = options.chunkDuration || 300; // 5 minutes
    this.overlapDuration = options.overlapDuration || 30; // 30 seconds
    this.outputDir = options.outputDir || './transcription_output';
    this.tempDir = path.join(this.outputDir, 'temp');
  }

  async initialize() {
    // Create output directories
    await fs.mkdir(this.outputDir, { recursive: true });
    await fs.mkdir(this.tempDir, { recursive: true });
    
    console.log(chalk.blue('🎙️  Meeting Transcriber CLI'));
    console.log(chalk.gray(`Model: ${this.modelType} | Speakers: ${this.speakerCount}`));
    console.log('');
  }

  async processAudio(inputPath) {
    const spinner = ora('Processing audio file...').start();
    
    try {
      // Validate input file
      const stats = await fs.stat(inputPath);
      const fileSizeMB = (stats.size / 1024 / 1024).toFixed(2);
      spinner.text = `Processing ${fileSizeMB}MB audio file...`;

      // Get audio duration
      const duration = await this.getAudioDuration(inputPath);
      console.log(chalk.green(`✓ Audio Duration: ${Math.floor(duration / 60)}:${(duration % 60).toFixed(0).padStart(2, '0')}`));

      // Convert to WAV and chunk if necessary
      const processedChunks = await this.prepareAudioChunks(inputPath, duration);
      spinner.succeed(`Audio processed into ${processedChunks.length} chunks`);

      return { chunks: processedChunks, duration };

    } catch (error) {
      spinner.fail('Audio processing failed');
      throw error;
    }
  }

  async getAudioDuration(inputPath) {
    return new Promise((resolve, reject) => {
      ffmpeg.ffprobe(inputPath, (err, metadata) => {
        if (err) reject(err);
        else resolve(metadata.format.duration);
      });
    });
  }

  async prepareAudioChunks(inputPath, duration) {
    const chunks = [];
    
    if (duration <= this.chunkDuration) {
      // Single chunk - just convert to WAV
      const outputPath = path.join(this.tempDir, 'chunk_0.wav');
      await this.convertToWav(inputPath, outputPath);
      chunks.push({
        path: outputPath,
        startTime: 0,
        endTime: duration,
        index: 0
      });
    } else {
      // Multiple chunks with overlap
      const numChunks = Math.ceil(duration / this.chunkDuration);
      
      for (let i = 0; i < numChunks; i++) {
        const startTime = Math.max(0, i * this.chunkDuration - (i > 0 ? this.overlapDuration : 0));
        const endTime = Math.min(duration, (i + 1) * this.chunkDuration);
        const outputPath = path.join(this.tempDir, `chunk_${i}.wav`);
        
        await this.extractChunk(inputPath, outputPath, startTime, endTime - startTime);
        
        chunks.push({
          path: outputPath,
          startTime,
          endTime,
          index: i,
          hasOverlap: i > 0
        });
      }
    }

    return chunks;
  }

  async convertToWav(inputPath, outputPath) {
    return new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .audioChannels(1)
        .audioFrequency(16000)
        .format('wav')
        .save(outputPath)
        .on('end', resolve)
        .on('error', reject);
    });
  }

  async extractChunk(inputPath, outputPath, startTime, duration) {
    return new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .seekInput(startTime)
        .duration(duration)
        .audioChannels(1)
        .audioFrequency(16000)
        .format('wav')
        .save(outputPath)
        .on('end', resolve)
        .on('error', reject);
    });
  }

  async transcribeChunks(chunks, originalAudioPath) {
    const spinner = ora('Processing audio with faster-whisper and pyannote...').start();
    const results = [];

    try {
      // Process each chunk
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        spinner.text = `Processing chunk ${i + 1}/${chunks.length} - Transcribing...`;
        
        // Transcribe with faster-whisper
        const transcription = await this.transcribeChunk(chunk);
        
        spinner.text = `Processing chunk ${i + 1}/${chunks.length} - Speaker diarization...`;
        
        // Perform speaker diarization
        const diarization = await this.performSpeakerDiarization(chunk.path);
        
        // Align transcription with speaker labels
        const alignedSegments = this.alignTranscriptionAndDiarization(transcription, diarization);
        
        results.push({
          transcription,
          diarization,
          alignedSegments,
          chunkIndex: i,
          startTime: chunk.startTime,
          endTime: chunk.endTime
        });

        spinner.text = `Completed chunk ${i + 1}/${chunks.length}`;
      }

      spinner.succeed(`Processed ${chunks.length} chunks with faster-whisper + pyannote`);
      return results;

    } catch (error) {
      spinner.fail('Audio processing failed');
      throw error;
    }
  }

  async transcribeChunk(chunk) {
    return new Promise((resolve, reject) => {
      // Run faster-whisper transcription
      const whisperProcess = spawn('python3', [
        '-c',
        `
import sys
import json
from faster_whisper import WhisperModel

# Initialize model with M3 optimizations
model = WhisperModel("${this.modelType}", device="auto", compute_type="int8")

# Transcribe with word-level timestamps
segments, info = model.transcribe("${chunk.path}", 
                                 word_timestamps=True,
                                 vad_filter=True,
                                 vad_parameters=dict(min_silence_duration_ms=500))

# Format output
result = {
    "language": info.language,
    "duration": info.duration,
    "segments": []
}

for segment in segments:
    result["segments"].append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip(),
        "words": [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability} 
                 for w in segment.words] if segment.words else []
    })

print(json.dumps(result))
        `
      ]);

      let output = '';
      let error = '';

      whisperProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      whisperProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      whisperProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Whisper transcription failed: ${error}`));
          return;
        }

        try {
          const result = JSON.parse(output.trim());
          resolve(result);
        } catch (parseError) {
          reject(new Error(`Failed to parse Whisper output: ${parseError.message}`));
        }
      });
    });
  }

  async performSpeakerDiarization(audioPath) {
    return new Promise((resolve, reject) => {
      // Run pyannote.audio speaker diarization
      const diarizationProcess = spawn('python3', [
        '-c',
        `
import sys
import json
from pyannote.audio import Pipeline

# Initialize pretrained pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                   use_auth_token="YOUR_HF_TOKEN_HERE")

# Perform diarization
diarization = pipeline("${audioPath}", num_speakers=${this.speakerCount})

# Convert to JSON format
result = {
    "speakers": [],
    "segments": []
}

for turn, _, speaker in diarization.itertracks(yield_label=True):
    result["segments"].append({
        "start": turn.start,
        "end": turn.end,
        "speaker": speaker
    })

# Get unique speakers
speakers = sorted(list(set([seg["speaker"] for seg in result["segments"]])))
result["speakers"] = speakers

print(json.dumps(result))
        `
      ]);

      let output = '';
      let error = '';

      diarizationProcess.stdout.on('data', (data) => {
        output += data.toString();
      });

      diarizationProcess.stderr.on('data', (data) => {
        error += data.toString();
      });

      diarizationProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Speaker diarization failed: ${error}`));
          return;
        }

        try {
          const result = JSON.parse(output.trim());
          resolve(result);
        } catch (parseError) {
          reject(new Error(`Failed to parse diarization output: ${parseError.message}`));
        }
      });
    });
  }

  alignTranscriptionAndDiarization(transcription, diarization) {
    // Align transcribed segments with speaker segments
    const alignedSegments = [];

    for (const transSegment of transcription.segments) {
      // Find overlapping speaker segment
      const speakerSegment = diarization.segments.find(spk => 
        spk.start <= transSegment.start && spk.end >= transSegment.end
      );

      // If no perfect match, find segment with most overlap
      const overlapSegment = speakerSegment || diarization.segments.reduce((best, current) => {
        const currentOverlap = Math.min(transSegment.end, current.end) - Math.max(transSegment.start, current.start);
        const bestOverlap = best ? Math.min(transSegment.end, best.end) - Math.max(transSegment.start, best.start) : 0;
        return currentOverlap > bestOverlap ? current : best;
      }, null);

      alignedSegments.push({
        ...transSegment,
        speaker: overlapSegment ? overlapSegment.speaker : 'SPEAKER_UNKNOWN',
        speakerConfidence: overlapSegment ? this.calculateOverlap(transSegment, overlapSegment) : 0
      });
    }

    return alignedSegments;
  }

  calculateOverlap(seg1, seg2) {
    const overlap = Math.min(seg1.end, seg2.end) - Math.max(seg1.start, seg2.start);
    const seg1Duration = seg1.end - seg1.start;
    return Math.max(0, overlap / seg1Duration);
  }

  async mergeTranscriptionsWithSpeakers(results) {
    const spinner = ora('Merging chunks and clustering speakers...').start();

    try {
      const allSegments = [];
      const speakerEmbeddings = new Map(); // For cross-chunk speaker consistency
      
      // Collect all segments with global timestamps
      for (const result of results) {
        for (const segment of result.alignedSegments) {
          const globalSegment = {
            ...segment,
            start: segment.start + result.startTime,
            end: segment.end + result.startTime,
            chunkIndex: result.chunkIndex,
            originalSpeaker: segment.speaker
          };
          allSegments.push(globalSegment);
        }
      }

      // Sort by start time
      allSegments.sort((a, b) => a.start - b.start);

      // Cluster speakers across chunks
      const clusteredSegments = await this.clusterSpeakersAcrossChunks(allSegments);

      // Remove overlaps and consolidate
      const finalSegments = this.removeOverlaps(clusteredSegments);

      spinner.succeed('Transcriptions merged and speakers clustered');
      return finalSegments;

    } catch (error) {
      spinner.fail('Merging failed');
      throw error;
    }
  }

  async clusterSpeakersAcrossChunks(segments) {
    // Simple speaker clustering based on temporal consistency
    // In production, you'd use speaker embeddings for better accuracy
    
    const speakerMap = new Map();
    const clusteredSegments = [];
    let globalSpeakerCounter = 0;

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const key = `${segment.chunkIndex}_${segment.originalSpeaker}`;
      
      if (!speakerMap.has(key)) {
        // Check if this speaker appears to be the same as a previous one
        // Based on temporal proximity and speaking patterns
        const similarSpeaker = this.findSimilarSpeaker(segment, clusteredSegments.slice(-10)); // Check last 10 segments
        
        if (similarSpeaker !== null) {
          speakerMap.set(key, similarSpeaker);
        } else {
          speakerMap.set(key, globalSpeakerCounter++);
        }
      }

      clusteredSegments.push({
        ...segment,
        globalSpeaker: speakerMap.get(key)
      });
    }

    return clusteredSegments;
  }

  findSimilarSpeaker(currentSegment, recentSegments) {
    // Simple heuristic: if a speaker was active recently (within 30 seconds)
    // and there's a gap suggesting turn-taking, it's likely the same speaker
    const timeThreshold = 30; // seconds
    
    for (let i = recentSegments.length - 1; i >= 0; i--) {
      const recentSegment = recentSegments[i];
      const timeDiff = currentSegment.start - recentSegment.end;
      
      if (timeDiff > timeThreshold) break;
      
      // If there's a reasonable gap (turn-taking) and similar context
      if (timeDiff > 1 && timeDiff < timeThreshold) {
        return recentSegment.globalSpeaker;
      }
    }
    
    return null;
  }

  removeOverlaps(segments) {
    // Simple overlap removal - in practice you'd want more sophisticated logic
    const filtered = [];
    let lastEnd = 0;

    for (const segment of segments) {
      if (segment.start >= lastEnd) {
        filtered.push(segment);
        lastEnd = segment.end;
      }
    }

    return filtered;
  }

  async generateOutput(segments, outputPath) {
    const spinner = ora('Generating output files...').start();

    try {
      // Generate transcript
      const transcript = this.formatTranscript(segments);
      
      // Write files
      await fs.writeFile(`${outputPath}.txt`, transcript);
      await fs.writeFile(`${outputPath}.json`, JSON.stringify({
        metadata: {
          model: this.modelType,
          speakerCount: this.speakerCount,
          generatedAt: new Date().toISOString()
        },
        segments
      }, null, 2));

      spinner.succeed('Output files generated');
      
      console.log(chalk.green('\n📝 Transcription Complete!'));
      console.log(chalk.gray(`Text: ${outputPath}.txt`));
      console.log(chalk.gray(`JSON: ${outputPath}.json`));

    } catch (error) {
      spinner.fail('Output generation failed');
      throw error;
    }
  }

  formatTranscript(segments) {
    let transcript = '# Meeting Transcript\n\n';
    transcript += `Generated: ${new Date().toLocaleString()}\n`;
    transcript += `Model: ${this.modelType}\n`;
    transcript += `Speakers: ${this.speakerCount}\n\n---\n\n`;

    for (const segment of segments) {
      const timestamp = this.formatTime(segment.start);
      const speaker = `Speaker ${segment.globalSpeaker + 1}`;
      transcript += `**${speaker}** [${timestamp}]: ${segment.text}\n\n`;
    }

    return transcript;
  }

  formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }

  async cleanup() {
    // Remove temporary files
    try {
      await fs.rmdir(this.tempDir, { recursive: true });
    } catch (error) {
      console.warn(chalk.yellow('Warning: Could not clean up temporary files'));
    }
  }

  async transcribe(inputPath) {
    try {
      await this.initialize();
      
      // Process audio
      const { chunks } = await this.processAudio(inputPath);
      
      // Transcribe chunks with speaker diarization
      const results = await this.transcribeChunks(chunks, inputPath);
      
      // Merge and cluster speakers across chunks
      const segments = await this.mergeTranscriptionsWithSpeakers(results);
      
      // Generate output
      const baseName = path.basename(inputPath, path.extname(inputPath));
      const outputPath = path.join(this.outputDir, baseName);
      await this.generateOutput(segments, outputPath);
      
      // Cleanup
      await this.cleanup();
      
      return outputPath;

    } catch (error) {
      console.error(chalk.red('\n❌ Transcription failed:'), error.message);
      await this.cleanup();
      process.exit(1);
    }
  }
}

// CLI Setup
program
  .name('meeting-transcriber')
  .description('AI-powered meeting transcription tool optimized for M3 MacBooks')
  .version('1.0.0');

program
  .command('transcribe <file>')
  .description('Transcribe an M4A audio file')
  .option('-m, --model <type>', 'Transcription model to use', 'whisper-base')
  .option('-s, --speakers <count>', 'Number of speakers', '2')
  .option('-o, --output <dir>', 'Output directory', './transcription_output')
  .action(async (file, options) => {
    const transcriber = new MeetingTranscriber({
      model: options.model,
      speakers: parseInt(options.speakers),
      outputDir: options.output
    });

    await transcriber.transcribe(file);
  });

program
  .command('models')
  .description('List available transcription models')
  .action(() => {
    console.log(chalk.blue('Available Models:'));
    console.log('  whisper-tiny   - Fastest, lower accuracy');
    console.log('  whisper-base   - Balanced speed/accuracy (default)');
    console.log('  whisper-small  - Better accuracy, slower');
    console.log('  whisper-medium - High accuracy, slower');
    console.log('  whisper-large  - Best accuracy, slowest');
  });

if (require.main === module) {
  program.parse();
}

module.exports = MeetingTranscriber;