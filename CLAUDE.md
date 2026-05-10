# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Car Noise Diagnosis System - a real-time audio analysis application that uses Google's YAMNet model with a custom-trained binary classifier to diagnose car engine problems. Designed for both desktop and Raspberry Pi deployment.

## Commands

### Setup
```bash
# Python 3.10 required
pip install -r requirements.txt

# Download YAMNet model (required)
curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite
```

### Running
```bash
python main.py                  # Interactive diagnosis system
python main.py --debug          # Debug mode with detailed YAMNet predictions
python main.py --compare        # Side-by-side Baseline vs Custom model comparison
```

### Training
```bash
python train_two_class.py           # Train binary classifier (Normal vs Knocking)
python train_yamnet_transfer.py     # YAMNet transfer learning
python scripts/tests/evaluate.py    # Model evaluation and comparison
```

### Testing
```bash
python scripts/tests/test_main_file_analysis.py   # File processing pipeline
python scripts/tests/test_realtime.py             # Real-time audio capture
python scripts/tests/test_verified_model.py       # Custom model verification
```

### Raspberry Pi
```bash
bash setup_pi.sh                # Automated setup (uses requirements-pi.txt)
```

## Architecture

```
Audio Input (Mic/File)
       ↓
AudioCapture / AudioFileLoader (src/audio/capture.py)
       ↓
MediaPipeAudioClassifier (src/models/mediapipe_classifier.py)
   ├─ YAMNet inference (521 audio classes)
   └─ Feature extraction: 88-dim vector
      • YAMNet Top-50 probabilities (50 dims)
      • Audio features: RMS, spectral centroid, rolloff (12 dims)
      • MFCC: 13 mean + 13 std (26 dims)
       ↓
CarNoiseDiagnoser (src/diagnosis/analyzer.py)
   ├─ Custom: Random Forest on extracted features
   └─ Baseline: YAMNet keywords → heuristic rules
       ↓
Korean diagnosis report
```

### Key Modules

- **main.py**: Entry point, CLI interaction, `CarNoiseDiagnosisSystem` class
- **src/audio/capture.py**: `AudioCapture` (real-time mic, ffmpeg-based), `AudioFileLoader` (WAV/MP3)
- **src/models/mediapipe_classifier.py**: `MediaPipeAudioClassifier` wraps YAMNet, extracts 88-dim features
- **src/diagnosis/analyzer.py**: `CarNoiseDiagnoser` with dual mode (Baseline/Custom), `CarPartStatus` enum

### Model Loading Priority

1. `car_classifier_binary.pkl` - Binary classifier (98.1% accuracy, current best)
2. `car_classifier_oneclass_verified.pkl`
3. `car_classifier_oneclass_v4.pkl`
4. Falls back to baseline YAMNet if no custom model available

## Data Pipeline

Training data location: `data/training/`
- `raw/`: Original data from YouTube/AudioSet
- `manual_review/`: Data validation pipeline
- `processed/`: Augmented training data

Data augmentation scripts:
- `augment_normal_sounds.py`
- `augment_knocking_sounds.py`

## Dependencies

Core: `mediapipe==0.10.18` (pinned for Raspberry Pi ARM compatibility), `librosa`, `scikit-learn`, `soundfile`

System: ffmpeg (required for audio capture)
