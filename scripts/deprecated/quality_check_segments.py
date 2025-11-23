#!/usr/bin/env python3
"""
Automatic Quality Check for Extracted Segments
추출된 노킹 세그먼트 자동 품질 검수

- 'Engine knocking' 점수 재확인
- 의심스러운 파일 삭제 또는 quarantine
"""

import sys
from pathlib import Path
from typing import List, Tuple
import shutil

sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader


def analyze_segment(segment_path: Path,
                    classifier: MediaPipeAudioClassifier) -> Tuple[float, str, List[dict]]:
    """
    Analyze a segment and return knocking score

    Returns:
        (knocking_score, top_category, top_5_predictions)
    """
    audio, sr = AudioFileLoader.load_audio(str(segment_path))
    results = classifier.classify_audio(audio, sr)
    top_5 = classifier.get_top_predictions(results, top_k=5)

    # Find 'Engine knocking' score
    knocking_score = 0.0
    top_category = top_5[0]['category_name'] if top_5 else "Unknown"

    for pred in top_5:
        if 'engine' in pred['category_name'].lower() and 'knock' in pred['category_name'].lower():
            knocking_score = pred['score']
            break

    return knocking_score, top_category, top_5


def main():
    """Main quality check process"""
    print("=" * 80)
    print("🔍 자동 품질 검수")
    print("=" * 80)

    # Paths
    segments_dir = Path("data/training/engine_knocking_segments_v2")
    quarantine_dir = Path("data/training/quarantine")  # 의심스러운 파일 격리

    if not segments_dir.exists():
        print(f"\n❌ {segments_dir}를 찾을 수 없습니다.")
        print("먼저 extract_knocking_segments_v2.py를 실행하세요.")
        return

    # Create quarantine directory
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Load YAMNet
    print(f"\n🤖 YAMNet 모델 로드 중...")
    classifier = MediaPipeAudioClassifier(
        model_path="data/models/yamnet.tflite",
        max_results=10,
        score_threshold=0.0
    )
    print("✅ 모델 로드 완료")

    # Get all segments
    segments = sorted(segments_dir.glob("*.wav"))
    print(f"\n📊 총 {len(segments)}개 세그먼트 검사 중...")

    # Categories
    high_quality = []      # knocking > 50%
    medium_quality = []    # knocking 30-50%
    low_quality = []       # knocking 10-30%
    suspicious = []        # knocking < 10% or not in top 5

    # Analyze each segment
    for i, segment in enumerate(segments, 1):
        knocking_score, top_category, top_5 = analyze_segment(segment, classifier)

        if i % 20 == 0:
            print(f"  진행: {i}/{len(segments)}")

        # Categorize
        if knocking_score >= 0.5:
            high_quality.append((segment, knocking_score, top_category))
        elif knocking_score >= 0.3:
            medium_quality.append((segment, knocking_score, top_category))
        elif knocking_score >= 0.1:
            low_quality.append((segment, knocking_score, top_category))
        else:
            suspicious.append((segment, knocking_score, top_category, top_5))

    # Summary
    print("\n" + "=" * 80)
    print("📊 품질 분석 결과")
    print("=" * 80)
    print(f"✅ 고품질 (>50%):       {len(high_quality):3}개")
    print(f"⚠️  중품질 (30-50%):     {len(medium_quality):3}개")
    print(f"⚠️  저품질 (10-30%):     {len(low_quality):3}개")
    print(f"❌ 의심 (<10%):         {len(suspicious):3}개")
    print(f"\n   총계:                {len(segments)}개")

    # Show suspicious files
    if suspicious:
        print("\n" + "=" * 80)
        print(f"❌ 의심스러운 파일 ({len(suspicious)}개):")
        print("=" * 80)
        for segment, score, top_cat, top_5 in suspicious[:10]:  # Show first 10
            print(f"\n📁 {segment.name}")
            print(f"   Engine knocking: {score:.1%}")
            print(f"   Top: {top_cat}")
            print("   Top 3:")
            for j, pred in enumerate(top_5[:3], 1):
                print(f"     {j}. {pred['category_name']:<30} {pred['score']:.1%}")

    # Action
    print("\n" + "=" * 80)
    print("🛠️  조치 옵션:")
    print("=" * 80)
    print(f"1. 의심 파일 격리 ({len(suspicious)}개)")
    print(f"2. 저품질 파일도 격리 ({len(suspicious) + len(low_quality)}개)")
    print(f"3. 중품질 이하 모두 격리 ({len(suspicious) + len(low_quality) + len(medium_quality)}개)")
    print("4. 아무것도 안 함 (수동 검수)")

    choice = input("\n선택 (1-4): ").strip()

    files_to_quarantine = []
    if choice == '1':
        files_to_quarantine = [s[0] for s in suspicious]
    elif choice == '2':
        files_to_quarantine = [s[0] for s in suspicious] + [s[0] for s in low_quality]
    elif choice == '3':
        files_to_quarantine = [s[0] for s in suspicious] + [s[0] for s in low_quality] + [s[0] for s in medium_quality]
    elif choice == '4':
        print("\n✅ 수동 검수를 진행하세요.")
        return
    else:
        print("❌ 잘못된 선택입니다.")
        return

    # Move to quarantine
    if files_to_quarantine:
        print(f"\n📦 {len(files_to_quarantine)}개 파일 격리 중...")
        for file in files_to_quarantine:
            dest = quarantine_dir / file.name
            shutil.move(str(file), str(dest))
        print(f"✅ 격리 완료: {quarantine_dir}")

        remaining = len(segments) - len(files_to_quarantine)
        print(f"\n📊 최종 결과:")
        print(f"   - 유지: {remaining}개 (고품질 학습 데이터)")
        print(f"   - 격리: {len(files_to_quarantine)}개")

        print(f"\n💡 다음 단계:")
        print(f"   python train_one_class.py  # 개선된 데이터로 재학습")

    print("=" * 80)


if __name__ == "__main__":
    main()
