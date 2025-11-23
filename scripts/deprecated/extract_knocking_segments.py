#!/usr/bin/env python3
"""
Extract knocking segments from audio files using YAMNet classification

Usage:
    python extract_knocking_segments.py
"""

import sys
import numpy as np
from pathlib import Path
from typing import List, Tuple
import soundfile as sf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


# YAMNet 키워드: 노킹 관련 소리
KNOCKING_KEYWORDS = [
    'knock', 'knocking', 'tap', 'tapping',
    'rattle', 'rattling', 'clank', 'clanking',
    'mechanical', 'engine', 'motor',
    'clatter', 'bang', 'thump'
]


def calculate_knocking_score(classifications: List[dict]) -> float:
    """Calculate knocking relevance score from YAMNet results

    Args:
        classifications: YAMNet classification results

    Returns:
        Score between 0 and 1 (higher = more likely knocking)
    """
    max_score = 0.0

    for result in classifications:
        for category in result.get('categories', []):
            category_name = category['category_name'].lower()
            score = category['score']

            # Check if any knocking keyword matches
            for keyword in KNOCKING_KEYWORDS:
                if keyword in category_name:
                    max_score = max(max_score, score)
                    break

    return max_score


def extract_segments(audio_path: Path,
                     classifier: MediaPipeAudioClassifier,
                     output_dir: Path,
                     window_size: float = 3.0,
                     hop_size: float = 1.5,
                     threshold: float = 0.3) -> int:
    """Extract knocking segments from an audio file

    Args:
        audio_path: Path to input audio file
        classifier: YAMNet classifier
        output_dir: Directory to save extracted segments
        window_size: Window size in seconds (default: 3.0)
        hop_size: Hop size in seconds (default: 1.5)
        threshold: Minimum knocking score to save segment (default: 0.3)

    Returns:
        Number of segments extracted
    """
    print(f"\n📂 Processing: {audio_path.name}")

    # Load audio
    try:
        audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_path))
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        return 0

    duration = len(audio_data) / sample_rate
    print(f"  ⏱️  Duration: {duration:.1f}s, Sample Rate: {sample_rate}Hz")

    # Calculate window parameters
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    segments_saved = 0
    base_name = audio_path.stem  # e.g., "knocking_01"

    # Sliding window
    for i, start_sample in enumerate(range(0, len(audio_data) - window_samples, hop_samples)):
        end_sample = start_sample + window_samples
        segment = audio_data[start_sample:end_sample]

        # Skip if too quiet
        rms = np.sqrt(np.mean(segment ** 2))
        if rms < 0.01:
            continue

        # Classify with YAMNet
        try:
            classifications = classifier.classify_audio(segment, sample_rate)
            knocking_score = calculate_knocking_score(classifications)

            # Get top prediction for logging
            top_pred = ""
            if classifications and classifications[0].get('categories'):
                top_cat = classifications[0]['categories'][0]
                top_pred = f"{top_cat['category_name']} ({top_cat['score']:.2f})"

            # Save if score exceeds threshold
            if knocking_score >= threshold:
                output_path = output_dir / f"{base_name}_seg_{segments_saved:03d}.wav"
                sf.write(str(output_path), segment, sample_rate)

                print(f"  ✅ Segment {segments_saved}: {start_sample/sample_rate:.1f}s-{end_sample/sample_rate:.1f}s "
                      f"(score: {knocking_score:.2f}, top: {top_pred})")
                segments_saved += 1
            else:
                # Optionally log skipped segments
                if i % 10 == 0:  # Log every 10th skip
                    print(f"  ⏭️  Skip {start_sample/sample_rate:.1f}s (score: {knocking_score:.2f}, top: {top_pred})")

        except Exception as e:
            print(f"  ⚠️  Classification error at {start_sample/sample_rate:.1f}s: {e}")
            continue

    print(f"  💾 Saved {segments_saved} segments")
    return segments_saved


def main():
    """Main extraction process"""
    print("=" * 80)
    print("🔊 YAMNet 기반 노킹 구간 자동 추출")
    print("=" * 80)

    # Paths
    input_dir = Path("data/training/engine_knocking")
    output_dir = Path("data/training/engine_knocking_segments")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")

    # Check YAMNet model
    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        print("다음 명령으로 다운로드하세요:")
        print("curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite")
        return

    # Initialize YAMNet classifier
    print(f"\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Find all WAV files
    audio_files = sorted(input_dir.glob("*.wav"))

    if not audio_files:
        print(f"\n❌ {input_dir}에서 WAV 파일을 찾을 수 없습니다")
        return

    print(f"\n📊 총 {len(audio_files)}개 파일 발견")

    # Process each file
    total_segments = 0
    for audio_file in audio_files:
        segments = extract_segments(
            audio_path=audio_file,
            classifier=classifier,
            output_dir=output_dir,
            window_size=3.0,      # 3초 윈도우
            hop_size=1.5,         # 1.5초씩 이동 (50% overlap)
            threshold=0.3         # 30% 이상 신뢰도
        )
        total_segments += segments

    # Summary
    print("\n" + "=" * 80)
    print("✅ 추출 완료!")
    print("=" * 80)
    print(f"📊 통계:")
    print(f"   - 원본 파일: {len(audio_files)}개")
    print(f"   - 추출된 노킹 구간: {total_segments}개")
    print(f"   - 저장 위치: {output_dir}")
    print(f"\n💡 다음 단계:")
    print(f"   1. {output_dir}에서 추출된 파일 확인")
    print(f"   2. 잘못 추출된 파일이 있다면 수동으로 삭제")
    print(f"   3. python train.py 실행하여 모델 학습")
    print("=" * 80)


if __name__ == "__main__":
    main()
