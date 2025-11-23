#!/usr/bin/env python3
"""
Test: Normal Engine Sound vs Knocking Detection
정상 엔진 소리와 노킹 소리를 비교 테스트
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from audio.capture import AudioCapture
from models.mediapipe_classifier import MediaPipeAudioClassifier
from diagnosis.analyzer import CarNoiseDiagnoser


def test_audio_sample(duration: float = 5.0):
    """오디오 샘플 테스트"""

    # Initialize
    yamnet_path = Path("data/models/yamnet.tflite")
    oneclass_path = Path("data/models/car_classifier_oneclass.pkl")

    print("🤖 모델 로딩 중...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_path),
        max_results=10,
        score_threshold=0.0
    )

    diagnoser = CarNoiseDiagnoser(model_path=str(oneclass_path))
    audio_capture = AudioCapture()

    print("✅ 모델 로딩 완료\n")

    # Capture audio
    print(f"🎙️  {duration}초간 오디오 녹음 중...")
    print("   (YouTube에서 소리를 재생하세요)")
    time.sleep(1)

    audio_capture.start_recording()
    audio_buffer = audio_capture.get_audio_buffer(duration)
    audio_capture.stop_recording()

    if len(audio_buffer) == 0:
        print("❌ 오디오 데이터가 없습니다.")
        return

    print("✅ 녹음 완료\n")

    # Analyze with YAMNet
    print("=" * 70)
    print("🤖 YAMNet 분석 (범용 오디오 분류기)")
    print("=" * 70)

    yamnet_results = classifier.classify_audio(audio_buffer, audio_capture.sample_rate)
    top_10 = classifier.get_top_predictions(yamnet_results, top_k=10)

    for i, pred in enumerate(top_10, 1):
        emoji = "🔴" if i <= 3 else "  "
        print(f"{emoji} {i:2}. {pred['category_name']:<35} {pred['score']:.1%}")

    # Extract embedding for One-Class
    print("\n" + "=" * 70)
    print("🎯 One-Class 모델 분석 (노킹 전용)")
    print("=" * 70)

    embedding = classifier.extract_embedding(audio_buffer, audio_capture.sample_rate)

    diagnosis = diagnoser.diagnose(
        {},
        yamnet_results,
        embedding=embedding,
        comparison_mode=False
    )

    # Results
    confidence = diagnosis['confidence']

    if diagnosis['issues']:
        print(f"\n🚨 결과: 엔진 노킹 감지! (신뢰도: {confidence:.0%})")
        for issue in diagnosis['issues']:
            print(f"   └─ {issue['description']}")

        if 'anomaly_score' in diagnosis and diagnosis['anomaly_score'] is not None:
            score = diagnosis['anomaly_score']
            print(f"\n📈 이상치 점수: {score:.3f}")
            print(f"   (양수 = 노킹 패턴, 음수 = 정상 패턴)")
    else:
        print(f"\n✅ 결과: 정상 (신뢰도: {confidence:.0%})")

        if 'anomaly_score' in diagnosis and diagnosis['anomaly_score'] is not None:
            score = diagnosis['anomaly_score']
            print(f"\n📈 이상치 점수: {score:.3f}")
            print(f"   (양수 = 노킹 패턴, 음수 = 정상 패턴)")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("🧪 정상 엔진 vs 노킹 비교 테스트")
    print("=" * 70)
    print("\n📋 테스트 방법:")
    print("1. YouTube에서 테스트할 엔진 소리 영상을 찾으세요")
    print("2. 스피커 볼륨을 적절히 조절하세요")
    print("3. 준비되면 Enter를 누르세요")
    print("\n💡 추천 영상:")
    print("   정상: https://youtube.com/watch?v=... (car idle sound)")
    print("   노킹: youtube_links.txt의 영상들")
    print("\n" + "-" * 70)

    while True:
        input("\n▶️  Enter를 눌러 테스트 시작 (종료: Ctrl+C): ")

        print("\n")
        test_audio_sample(duration=5.0)

        print("\n" + "-" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 테스트 종료")
    except Exception as e:
        print(f"\n❌ 오류: {e}")
