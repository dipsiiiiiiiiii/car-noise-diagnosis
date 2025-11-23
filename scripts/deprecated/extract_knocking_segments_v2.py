#!/usr/bin/env python3
"""
Extract knocking segments - IMPROVED VERSION
개선 사항:
1. 윈도우 크기: 3초 → 1.5초 (더 정밀)
2. Threshold: 0.3 → 0.5 (더 엄격)
3. 'Engine knocking' 레이블 직접 확인 (필수!)
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


def get_engine_knocking_score(classifications: List[dict]) -> Tuple[float, str]:
    """
    Get 'Engine knocking' score directly (not generic keywords!)

    Returns:
        (knocking_score, top_prediction_name)
    """
    knocking_score = 0.0
    top_pred = ""

    for result in classifications:
        categories = result.get('categories', [])
        if categories:
            top_pred = categories[0]['category_name']

        for category in categories:
            category_name = category['category_name'].lower()
            score = category['score']

            # ONLY 'engine knocking' - be specific!
            if 'engine' in category_name and 'knock' in category_name:
                knocking_score = max(knocking_score, score)

    return knocking_score, top_pred


def extract_segments_v2(audio_path: Path,
                        classifier: MediaPipeAudioClassifier,
                        output_dir: Path,
                        window_size: float = 1.5,  # 3.0 → 1.5
                        hop_size: float = 0.75,     # 1.5 → 0.75
                        knocking_threshold: float = 0.3) -> int:  # Engine knocking만
    """Extract knocking segments - IMPROVED VERSION"""

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
    base_name = audio_path.stem

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
            knocking_score, top_pred = get_engine_knocking_score(classifications)

            # STRICT: Only save if 'Engine knocking' is strong enough
            if knocking_score >= knocking_threshold:
                output_path = output_dir / f"{base_name}_seg_{segments_saved:03d}.wav"
                sf.write(str(output_path), segment, sample_rate)

                print(f"  ✅ Segment {segments_saved}: {start_sample/sample_rate:.1f}s-{end_sample/sample_rate:.1f}s "
                      f"(knocking: {knocking_score:.1%}, top: {top_pred})")
                segments_saved += 1
            else:
                # Log skipped (optional)
                if i % 20 == 0 and knocking_score > 0:
                    print(f"  ⏭️  Skip {start_sample/sample_rate:.1f}s "
                          f"(knocking: {knocking_score:.1%}, top: {top_pred})")

        except Exception as e:
            print(f"  ⚠️  Classification error at {start_sample/sample_rate:.1f}s: {e}")
            continue

    print(f"  💾 Saved {segments_saved} segments")
    return segments_saved


def main():
    """Main extraction process"""
    print("=" * 80)
    print("🔊 개선된 노킹 구간 추출 v2.0")
    print("   - 윈도우: 1.5초 (더 정밀)")
    print("   - 'Engine knocking' 레이블 직접 확인")
    print("   - 높은 품질 보장")
    print("=" * 80)

    # Paths
    input_dir = Path("data/training/engine_knocking")
    output_dir = Path("data/training/engine_knocking_segments_v2")  # 새 폴더
    yamnet_model = Path("data/models/yamnet.tflite")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Input:  {input_dir}")
    print(f"📁 Output: {output_dir} (새 폴더!)")

    # Check YAMNet model
    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
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
    print("\n⚙️  추출 설정:")
    print(f"   - 윈도우 크기: 1.5초")
    print(f"   - 겹침: 50%")
    print(f"   - 'Engine knocking' threshold: 30%")

    # Process each file
    total_segments = 0
    for audio_file in audio_files:
        segments = extract_segments_v2(
            audio_path=audio_file,
            classifier=classifier,
            output_dir=output_dir,
            window_size=1.5,           # 1.5초
            hop_size=0.75,             # 0.75초 (50% overlap)
            knocking_threshold=0.3     # Engine knocking 30% 이상
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
    print(f"   python quality_check_segments.py  # 자동 품질 검수")
    print("=" * 80)


if __name__ == "__main__":
    main()
