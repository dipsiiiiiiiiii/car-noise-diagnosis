#!/usr/bin/env python3
"""
Test One-Class model with various sounds
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from audio.capture import AudioFileLoader
from models.mediapipe_classifier import MediaPipeAudioClassifier
from diagnosis.analyzer import CarNoiseDiagnoser


def test_audio_file(file_path: str, description: str):
    """Test a single audio file"""
    print("\n" + "=" * 80)
    print(f"🎵 테스트: {description}")
    print(f"📁 파일: {file_path}")
    print("=" * 80)

    # Load audio
    try:
        audio_data, sample_rate = AudioFileLoader.load_audio(file_path)
        print(f"✅ 오디오 로드: {len(audio_data)} 샘플, {sample_rate}Hz")
    except Exception as e:
        print(f"❌ 오디오 로드 실패: {e}")
        return

    # Initialize models
    yamnet_path = "data/models/yamnet.tflite"
    oneclass_path = "data/models/car_classifier_oneclass.pkl"

    classifier = MediaPipeAudioClassifier(
        model_path=yamnet_path,
        max_results=10,
        score_threshold=0.0
    )

    diagnoser = CarNoiseDiagnoser(model_path=oneclass_path)

    # Classify with YAMNet
    mediapipe_results = classifier.classify_audio(audio_data, sample_rate)

    # Extract embedding
    embedding = classifier.extract_embedding(audio_data, sample_rate)

    if embedding is None:
        print("❌ 임베딩 추출 실패")
        return

    # Diagnose
    diagnosis = diagnoser.diagnose(
        audio_features={},
        mediapipe_results=mediapipe_results,
        embedding=embedding
    )

    # Print results
    print("\n📊 진단 결과:")
    print(f"   모드: {diagnosis['mode']}")
    print(f"   상태: {diagnosis['overall_status'].value if hasattr(diagnosis['overall_status'], 'value') else diagnosis['overall_status']}")
    print(f"   예측: {diagnosis['prediction']} (0=정상, 1=노킹)")
    print(f"   신뢰도: {diagnosis['confidence']:.1%}")

    if diagnosis.get('anomaly_score') is not None:
        print(f"   이상치 점수: {diagnosis['anomaly_score']:.4f} (낮을수록 이상)")

    if diagnosis['issues']:
        print("\n⚠️  감지된 문제:")
        for issue in diagnosis['issues']:
            print(f"   - {issue['description']}")

    print("\n💡 권장사항:")
    for rec in diagnosis['recommendations']:
        print(f"   {rec}")

    # Show top YAMNet detections
    print("\n🔊 YAMNet 탑 5 감지:")
    if mediapipe_results and mediapipe_results[0].get('categories'):
        for i, cat in enumerate(mediapipe_results[0]['categories'][:5], 1):
            print(f"   {i}. {cat['category_name']}: {cat['score']:.1%}")


def main():
    print("=" * 80)
    print("🧪 One-Class 모델 테스트")
    print("=" * 80)

    # Test 1: Knocking sample (should detect as anomaly)
    knocking_file = "data/training/engine_knocking_segments/knocking_01_seg_000.wav"
    if Path(knocking_file).exists():
        test_audio_file(knocking_file, "엔진 노킹 샘플 (감지되어야 함)")

    # Test 2: Another knocking sample
    knocking_file2 = "data/training/engine_knocking_segments/knocking_15_seg_005.wav"
    if Path(knocking_file2).exists():
        test_audio_file(knocking_file2, "엔진 노킹 샘플 #2 (감지되어야 함)")

    # Print summary
    print("\n" + "=" * 80)
    print("✅ 테스트 완료!")
    print("=" * 80)
    print("\n💡 추가 테스트:")
    print("   - 음악 파일을 테스트하려면: python test_oneclass.py <음악파일.mp3>")
    print("   - 말소리 파일을 테스트하려면: python test_oneclass.py <말소리.wav>")
    print("   - 정상 엔진 소리를 테스트하려면: python test_oneclass.py <정상엔진.wav>")


if __name__ == "__main__":
    # If file path provided as argument, test that file
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        description = "사용자 제공 파일"
        test_audio_file(file_path, description)
    else:
        # Run default tests
        main()
