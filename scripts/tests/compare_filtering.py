#!/usr/bin/env python3
"""
Compare audio filtering impact on diagnosis accuracy

Tests whether voice activity detection + filtering improves results
or is unnecessary overhead.

Usage:
    python compare_filtering.py
    python compare_filtering.py --data-dir data/training --samples 10
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier, AudioPreprocessor
from audio.capture import AudioFileLoader


def analyze_with_filtering(audio_data: np.ndarray, sample_rate: int,
                           classifier: MediaPipeAudioClassifier,
                           preprocessor: AudioPreprocessor) -> Dict:
    """Analyze with voice detection + filtering (current approach)"""

    start_time = time.time()

    # 1. Voice activity detection
    voice_analysis = preprocessor.detect_voice_activity(audio_data, sample_rate)

    # 2. Apply filtering if voice detected
    if voice_analysis['voice_detected']:
        filtered_audio = preprocessor.filter_background_noise(audio_data, sample_rate)
        analysis_audio = filtered_audio
    else:
        analysis_audio = audio_data

    # 3. YAMNet classification
    mediapipe_results = classifier.classify_audio(analysis_audio, sample_rate)

    # 4. Extract features
    audio_features = preprocessor.extract_features(analysis_audio, sample_rate)
    engine_patterns = preprocessor.detect_engine_patterns(analysis_audio, sample_rate)

    processing_time = time.time() - start_time

    return {
        'mediapipe_results': mediapipe_results,
        'audio_features': audio_features,
        'engine_patterns': engine_patterns,
        'voice_detected': voice_analysis['voice_detected'],
        'filtering_applied': voice_analysis['voice_detected'],
        'processing_time': processing_time
    }


def analyze_without_filtering(audio_data: np.ndarray, sample_rate: int,
                               classifier: MediaPipeAudioClassifier,
                               preprocessor: AudioPreprocessor) -> Dict:
    """Analyze without filtering (simple approach)"""

    start_time = time.time()

    # Direct YAMNet classification
    mediapipe_results = classifier.classify_audio(audio_data, sample_rate)

    # Extract features
    audio_features = preprocessor.extract_features(audio_data, sample_rate)
    engine_patterns = preprocessor.detect_engine_patterns(audio_data, sample_rate)

    processing_time = time.time() - start_time

    return {
        'mediapipe_results': mediapipe_results,
        'audio_features': audio_features,
        'engine_patterns': engine_patterns,
        'voice_detected': None,
        'filtering_applied': False,
        'processing_time': processing_time
    }


def compare_results(with_filter: Dict, without_filter: Dict, classifier: MediaPipeAudioClassifier) -> Dict:
    """Compare two analysis results"""

    # Get top predictions
    top_with = classifier.get_top_predictions(with_filter['mediapipe_results'], top_k=10)
    top_without = classifier.get_top_predictions(without_filter['mediapipe_results'], top_k=10)

    # Get vehicle sounds
    vehicle_with = classifier.filter_vehicle_sounds(with_filter['mediapipe_results'])
    vehicle_without = classifier.filter_vehicle_sounds(without_filter['mediapipe_results'])

    # Calculate confidence scores
    confidence_with = np.mean([pred['score'] for pred in top_with[:5]]) if top_with else 0
    confidence_without = np.mean([pred['score'] for pred in top_without[:5]]) if top_without else 0

    # Check if top prediction changed
    top_pred_with = top_with[0]['category_name'] if top_with else "None"
    top_pred_without = top_without[0]['category_name'] if top_without else "None"
    prediction_changed = (top_pred_with != top_pred_without)

    return {
        'top_with': top_with[:5],
        'top_without': top_without[:5],
        'vehicle_with': vehicle_with[:3],
        'vehicle_without': vehicle_without[:3],
        'confidence_with': confidence_with,
        'confidence_without': confidence_without,
        'confidence_diff': confidence_with - confidence_without,
        'prediction_changed': prediction_changed,
        'top_pred_with': top_pred_with,
        'top_pred_without': top_pred_without,
        'time_with': with_filter['processing_time'],
        'time_without': without_filter['processing_time'],
        'filtering_applied': with_filter['filtering_applied']
    }


def run_comparison_test(data_dir: Path, classifier: MediaPipeAudioClassifier,
                        preprocessor: AudioPreprocessor, max_samples: int = 10) -> List[Dict]:
    """Run comparison test on multiple audio files"""

    print(f"\n🔬 음성 필터링 효과 검증 실험")
    print("=" * 70)
    print(f"데이터 디렉토리: {data_dir}")
    print(f"최대 샘플 수: {max_samples}")
    print("=" * 70)

    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a', '*.flac', '*.ogg']

    # Collect all audio files
    all_files = []
    for ext in audio_extensions:
        all_files.extend(data_dir.rglob(ext))

    if not all_files:
        print(f"❌ {data_dir}에서 오디오 파일을 찾을 수 없습니다.")
        return []

    # Limit samples
    import random
    random.seed(42)
    test_files = random.sample(all_files, min(len(all_files), max_samples))

    print(f"\n📁 테스트 파일: {len(test_files)}개")
    print("-" * 70)

    results = []

    for i, audio_file in enumerate(test_files, 1):
        print(f"\n[{i}/{len(test_files)}] {audio_file.name}")

        try:
            # Load audio
            audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_file))

            if len(audio_data) == 0:
                print("  ⚠️  빈 파일, 건너뛰기")
                continue

            # Analyze with filtering
            print("  🔧 필터링 있음 분석 중...")
            result_with = analyze_with_filtering(audio_data, sample_rate, classifier, preprocessor)

            # Analyze without filtering
            print("  ⚡ 필터링 없음 분석 중...")
            result_without = analyze_without_filtering(audio_data, sample_rate, classifier, preprocessor)

            # Compare
            comparison = compare_results(result_with, result_without, classifier)
            comparison['file_name'] = audio_file.name
            comparison['file_path'] = str(audio_file)

            results.append(comparison)

            # Print summary
            if comparison['filtering_applied']:
                print(f"  ✅ 필터링 적용됨 (음성 감지)")
            else:
                print(f"  ⚪ 필터링 미적용 (기계음만)")

            if comparison['prediction_changed']:
                print(f"  🔄 예측 변경: {comparison['top_pred_without']} → {comparison['top_pred_with']}")
            else:
                print(f"  ➡️  예측 동일: {comparison['top_pred_without']}")

            print(f"  📊 신뢰도: {comparison['confidence_without']:.1%} → {comparison['confidence_with']:.1%} "
                  f"({comparison['confidence_diff']:+.1%}p)")
            print(f"  ⏱️  처리시간: {comparison['time_without']:.2f}s → {comparison['time_with']:.2f}s")

        except Exception as e:
            print(f"  ❌ 오류: {e}")
            continue

    return results


def print_summary_statistics(results: List[Dict]):
    """Print summary statistics"""

    if not results:
        print("\n❌ 분석 결과가 없습니다.")
        return

    print("\n" + "=" * 70)
    print("📊 종합 통계")
    print("=" * 70)

    # Count filtering applied
    filtering_applied_count = sum(1 for r in results if r['filtering_applied'])
    print(f"\n1️⃣  필터링 적용 케이스: {filtering_applied_count}/{len(results)} "
          f"({filtering_applied_count/len(results)*100:.1f}%)")

    # Count prediction changes
    prediction_changed_count = sum(1 for r in results if r['prediction_changed'])
    print(f"2️⃣  예측 변경 케이스: {prediction_changed_count}/{len(results)} "
          f"({prediction_changed_count/len(results)*100:.1f}%)")

    # Average confidence difference
    avg_conf_diff = np.mean([r['confidence_diff'] for r in results])
    print(f"3️⃣  평균 신뢰도 변화: {avg_conf_diff:+.2%}p")

    # Confidence improvement cases
    improved = sum(1 for r in results if r['confidence_diff'] > 0.01)
    worsened = sum(1 for r in results if r['confidence_diff'] < -0.01)
    unchanged = len(results) - improved - worsened

    print(f"\n신뢰도 변화 분포:")
    print(f"  ✅ 개선: {improved}/{len(results)} ({improved/len(results)*100:.1f}%)")
    print(f"  ❌ 악화: {worsened}/{len(results)} ({worsened/len(results)*100:.1f}%)")
    print(f"  ➡️  동일: {unchanged}/{len(results)} ({unchanged/len(results)*100:.1f}%)")

    # Average processing time
    avg_time_with = np.mean([r['time_with'] for r in results])
    avg_time_without = np.mean([r['time_without'] for r in results])
    time_overhead = avg_time_with - avg_time_without

    print(f"\n4️⃣  처리 시간:")
    print(f"  필터링 없음: {avg_time_without:.3f}s")
    print(f"  필터링 있음: {avg_time_with:.3f}s")
    print(f"  오버헤드: +{time_overhead:.3f}s (+{time_overhead/avg_time_without*100:.1f}%)")

    print("\n" + "=" * 70)
    print("🎯 결론:")
    print("=" * 70)

    if avg_conf_diff > 0.02:
        print("✅ 필터링이 신뢰도를 유의미하게 개선합니다.")
        print("   → 필터링 유지 권장")
    elif avg_conf_diff < -0.02:
        print("❌ 필터링이 신뢰도를 저하시킵니다.")
        print("   → 필터링 제거 권장")
    else:
        print("➡️  필터링의 효과가 미미합니다.")
        if time_overhead > 0.1:
            print(f"   → 처리시간 {time_overhead:.2f}s 증가하므로 제거 고려")
        else:
            print("   → 현상 유지 또는 제거 모두 가능")

    if filtering_applied_count == 0:
        print("\n⚠️  주의: 테스트 데이터에서 음성이 감지되지 않았습니다.")
        print("   음성이 포함된 오디오로 다시 테스트하세요.")


def main():
    parser = argparse.ArgumentParser(description="Compare filtering effectiveness")
    parser.add_argument('--data-dir', type=str, default='data/training',
                        help='Directory containing audio files')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples to test (default: 10)')
    parser.add_argument('--yamnet', type=str, default='data/models/yamnet.tflite',
                        help='YAMNet model path')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    yamnet_path = Path(args.yamnet)

    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리 없음: {data_dir}")
        return 1

    if not yamnet_path.exists():
        print(f"❌ YAMNet 모델 없음: {yamnet_path}")
        return 1

    # Initialize
    print("🤖 YAMNet 로딩...")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_path),
        max_results=50,
        score_threshold=0.0
    )

    preprocessor = AudioPreprocessor()

    # Run comparison
    results = run_comparison_test(data_dir, classifier, preprocessor, args.samples)

    # Print statistics
    if results:
        print_summary_statistics(results)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
