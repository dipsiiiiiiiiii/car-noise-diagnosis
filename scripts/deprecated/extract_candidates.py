#!/usr/bin/env python3
"""
Extract Candidate Knocking Segments for Manual Review
YAMNet으로 1차 필터링 → 사용자 수동 검수
"""

import sys
import numpy as np
from pathlib import Path
import soundfile as sf

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def get_engine_knocking_score(classifications):
    """Get 'Engine knocking' score from YAMNet"""
    knocking_score = 0.0
    top_pred = ""

    for result in classifications:
        categories = result.get('categories', [])
        if categories:
            top_pred = categories[0]['category_name']

        for category in categories:
            category_name = category['category_name'].lower()
            score = category['score']

            if 'engine' in category_name and 'knock' in category_name:
                knocking_score = max(knocking_score, score)

    return knocking_score, top_pred


def extract_candidates(audio_path: Path,
                       classifier: MediaPipeAudioClassifier,
                       output_dir: Path,
                       window_size: float = 1.5,
                       hop_size: float = 0.75,
                       threshold: float = 0.2) -> int:
    """
    Extract candidate segments for manual review

    Args:
        threshold: YAMNet 'Engine knocking' 최소 점수 (낮게 설정해서 후보 많이 추출)
    """
    print(f"\n📂 처리 중: {audio_path.name}")

    # Load audio
    try:
        audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_path))
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return 0

    duration = len(audio_data) / sample_rate
    print(f"  ⏱️  길이: {duration:.1f}초")

    # Window parameters
    window_samples = int(window_size * sample_rate)
    hop_samples = int(hop_size * sample_rate)

    candidates_saved = 0
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

            # Save candidates (낮은 threshold로 후보 많이 수집)
            if knocking_score >= threshold:
                # 파일명에 시간, 점수, Top 분류 포함
                start_time = start_sample / sample_rate
                output_name = (f"{base_name}_"
                              f"t{start_time:06.1f}s_"
                              f"score{int(knocking_score*100):02d}_"
                              f"{candidates_saved:03d}.wav")

                output_path = output_dir / output_name
                sf.write(str(output_path), segment, sample_rate)

                print(f"  ✅ [{candidates_saved:3d}] {start_time:6.1f}s | "
                      f"노킹: {knocking_score:5.1%} | Top: {top_pred}")
                candidates_saved += 1

        except Exception as e:
            print(f"  ⚠️  분류 오류 at {start_sample/sample_rate:.1f}s: {e}")
            continue

    print(f"  💾 총 {candidates_saved}개 후보 추출")
    return candidates_saved


def main():
    print("="*80)
    print("🔍 후보 노킹 세그먼트 추출 (수동 검수용)")
    print("="*80)

    # Paths
    input_dir = Path("data/training/engine_knocking")
    output_dir = Path("data/training/manual_workflow/1_candidates")
    yamnet_model = Path("data/models/yamnet.tflite")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 입력:  {input_dir}")
    print(f"📁 출력:  {output_dir}")
    print("\n⚙️  설정:")
    print("   - 윈도우: 1.5초")
    print("   - Threshold: 20% (낮게 설정 - 후보 많이 수집)")
    print("   - 목적: 1차 필터링 후 수동 검수")

    # Check paths
    if not input_dir.exists():
        print(f"\n❌ 입력 폴더를 찾을 수 없습니다: {input_dir}")
        return

    if not yamnet_model.exists():
        print(f"\n❌ YAMNet 모델을 찾을 수 없습니다: {yamnet_model}")
        return

    # Load YAMNet
    print(f"\n🤖 YAMNet 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_model),
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Find audio files
    audio_files = sorted(input_dir.glob("*.wav"))
    audio_files.extend(sorted(input_dir.glob("*.mp3")))
    audio_files.extend(sorted(input_dir.glob("*.mp4")))

    if not audio_files:
        print(f"\n❌ {input_dir}에서 오디오 파일을 찾을 수 없습니다")
        return

    print(f"\n📊 총 {len(audio_files)}개 파일 발견")

    # Process each file
    total_candidates = 0
    for audio_file in audio_files:
        candidates = extract_candidates(
            audio_path=audio_file,
            classifier=classifier,
            output_dir=output_dir,
            window_size=1.5,
            hop_size=0.75,
            threshold=0.2  # 20% 이상이면 후보로 추출
        )
        total_candidates += candidates

    # Summary
    print("\n" + "="*80)
    print("✅ 1차 추출 완료!")
    print("="*80)
    print(f"📊 통계:")
    print(f"   - 원본 파일: {len(audio_files)}개")
    print(f"   - 추출된 후보: {total_candidates}개")
    print(f"   - 저장 위치: {output_dir}")
    print(f"\n💡 다음 단계:")
    print(f"   python review_segments.py  # 수동 검수 시작")
    print("="*80)


if __name__ == "__main__":
    main()
