#!/usr/bin/env python3
"""
Real-time audio filtering comparison

Shows side-by-side comparison of:
- WITHOUT filtering (left)
- WITH filtering (right)

Usage:
    python realtime_filter_comparison.py
    python realtime_filter_comparison.py --duration 3
"""

import sys
import argparse
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier, AudioPreprocessor
from audio.capture import AudioCapture


class RealtimeFilteringComparison:
    def __init__(self):
        # YAMNet 모델 경로
        yamnet_path = Path(__file__).parent / "data" / "models" / "yamnet.tflite"
        if not yamnet_path.exists():
            print(f"❌ YAMNet 모델이 없습니다: {yamnet_path}")
            print("다음 명령어로 다운로드하세요:")
            print("curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite")
            sys.exit(1)

        print("🤖 YAMNet 로딩 중...")
        self.classifier = MediaPipeAudioClassifier(
            model_path=str(yamnet_path),
            max_results=10,
            score_threshold=0.0
        )
        self.preprocessor = AudioPreprocessor()
        self.audio_capture = AudioCapture()
        print("✅ 초기화 완료")

    def analyze_without_filtering(self, audio_data: np.ndarray, sample_rate: int):
        """필터링 없이 분석"""
        start_time = time.time()

        # 바로 YAMNet 분류
        mediapipe_results = self.classifier.classify_audio(audio_data, sample_rate)
        top_predictions = self.classifier.get_top_predictions(mediapipe_results, top_k=5)
        vehicle_sounds = self.classifier.filter_vehicle_sounds(mediapipe_results)

        processing_time = time.time() - start_time

        return {
            'top_predictions': top_predictions,
            'vehicle_sounds': vehicle_sounds,
            'processing_time': processing_time,
            'voice_detected': None
        }

    def analyze_with_filtering(self, audio_data: np.ndarray, sample_rate: int):
        """필터링 포함 분석"""
        start_time = time.time()

        # 음성 감지
        voice_analysis = self.preprocessor.detect_voice_activity(audio_data, sample_rate)

        # 필터링 적용 여부
        if voice_analysis['voice_detected']:
            filtered_audio = self.preprocessor.filter_background_noise(audio_data, sample_rate)
            analysis_audio = filtered_audio
        else:
            analysis_audio = audio_data

        # YAMNet 분류
        mediapipe_results = self.classifier.classify_audio(analysis_audio, sample_rate)
        top_predictions = self.classifier.get_top_predictions(mediapipe_results, top_k=5)
        vehicle_sounds = self.classifier.filter_vehicle_sounds(mediapipe_results)

        processing_time = time.time() - start_time

        return {
            'top_predictions': top_predictions,
            'vehicle_sounds': vehicle_sounds,
            'processing_time': processing_time,
            'voice_detected': voice_analysis['voice_detected'],
            'audio_type': voice_analysis['audio_type']
        }

    def print_side_by_side_results(self, without_filter, with_filter, audio_rms):
        """양쪽 결과를 나란히 출력"""

        print("\n" + "="*100)
        print(f"🎙️  실시간 필터링 비교 (RMS: {audio_rms:.4f})")
        print("="*100)

        # 헤더
        left_header = "⚡ 필터링 없음 (단순)"
        right_header = "🔧 필터링 있음 (현재)"

        print(f"\n{left_header:<50} | {right_header}")
        print("-"*50 + " | " + "-"*50)

        # 처리 시간
        print(f"⏱️  처리시간: {without_filter['processing_time']:.3f}s{'':<34} | "
              f"⏱️  처리시간: {with_filter['processing_time']:.3f}s")

        # 음성 감지 여부
        voice_status_right = ""
        if with_filter['voice_detected'] is not None:
            if with_filter['voice_detected']:
                voice_status_right = f"🎤 음성 감지됨 → 필터링 적용"
            else:
                voice_status_right = f"🔧 기계음만 → 필터링 안 함"

        print(f"{'':<50} | {voice_status_right}")

        print("\n" + "="*100)
        print(f"{'🤖 YAMNet Top 5 예측':<50} | {'🤖 YAMNet Top 5 예측'}")
        print("="*100)

        # Top 5 predictions 나란히
        for i in range(5):
            left = without_filter['top_predictions'][i] if i < len(without_filter['top_predictions']) else None
            right = with_filter['top_predictions'][i] if i < len(with_filter['top_predictions']) else None

            left_str = f"  {i+1}. {left['category_name']:<30} {left['score']:.1%}" if left else ""
            right_str = f"  {i+1}. {right['category_name']:<30} {right['score']:.1%}" if right else ""

            # 예측이 다르면 표시
            marker = ""
            if left and right and left['category_name'] != right['category_name']:
                marker = " 🔄"

            print(f"{left_str:<50} | {right_str}{marker}")

        print("\n" + "="*100)
        print(f"{'🚗 차량 관련 소리 (Top 3)':<50} | {'🚗 차량 관련 소리 (Top 3)'}")
        print("="*100)

        # Vehicle sounds 나란히
        max_vehicles = max(len(without_filter['vehicle_sounds']), len(with_filter['vehicle_sounds']))
        max_vehicles = min(max_vehicles, 3)

        if max_vehicles == 0:
            print(f"{'  ❌ 차량 소리 없음':<50} | {'  ❌ 차량 소리 없음'}")
        else:
            for i in range(max_vehicles):
                left = without_filter['vehicle_sounds'][i] if i < len(without_filter['vehicle_sounds']) else None
                right = with_filter['vehicle_sounds'][i] if i < len(with_filter['vehicle_sounds']) else None

                left_str = f"  - {left['category_name']}: {left['score']:.1%}" if left else ""
                right_str = f"  - {right['category_name']}: {right['score']:.1%}" if right else ""

                print(f"{left_str:<50} | {right_str}")

        print("\n" + "="*100)

        # 차이 분석
        self._print_difference_analysis(without_filter, with_filter)

        print("="*100 + "\n")

    def _print_difference_analysis(self, without_filter, with_filter):
        """차이점 분석"""
        print("📊 차이 분석:")

        # Top 1 prediction 비교
        top_without = without_filter['top_predictions'][0] if without_filter['top_predictions'] else None
        top_with = with_filter['top_predictions'][0] if with_filter['top_predictions'] else None

        if top_without and top_with:
            if top_without['category_name'] == top_with['category_name']:
                print(f"  ✅ 1등 예측 동일: {top_without['category_name']}")

                # 신뢰도 비교
                conf_diff = (top_with['score'] - top_without['score']) * 100
                if abs(conf_diff) > 1:
                    if conf_diff > 0:
                        print(f"  📈 필터링 후 신뢰도 증가: {conf_diff:+.1f}%p")
                    else:
                        print(f"  📉 필터링 후 신뢰도 감소: {conf_diff:+.1f}%p")
                else:
                    print(f"  ➡️  신뢰도 거의 동일")
            else:
                print(f"  🔄 1등 예측 변경: {top_without['category_name']} → {top_with['category_name']}")
                print(f"     필터링 없음: {top_without['score']:.1%}")
                print(f"     필터링 있음: {top_with['score']:.1%}")

        # 차량 소리 개수 비교
        vehicle_count_without = len(without_filter['vehicle_sounds'])
        vehicle_count_with = len(with_filter['vehicle_sounds'])

        if vehicle_count_without != vehicle_count_with:
            print(f"  🚗 차량 소리 감지 수: {vehicle_count_without}개 → {vehicle_count_with}개")

        # 처리 시간 오버헤드
        time_overhead = with_filter['processing_time'] - without_filter['processing_time']
        overhead_pct = (time_overhead / without_filter['processing_time']) * 100

        if time_overhead > 0.05:
            print(f"  ⏱️  필터링 오버헤드: +{time_overhead:.3f}s (+{overhead_pct:.1f}%)")

    def run_continuous_comparison(self, duration: float = 3.0):
        """연속 비교 모드"""
        print("\n🔄 실시간 필터링 비교 모드")
        print("="*100)
        print(f"📌 {duration}초마다 양쪽 결과를 비교합니다")
        print("📌 Ctrl+C로 중단")
        print("📌 마이크에 소리를 들려주세요 (사람 말소리 + 기계음 등)")
        print("="*100)

        try:
            self.audio_capture.start_recording()

            while True:
                # 오디오 수집
                audio_buffer = self.audio_capture.get_audio_buffer(duration)

                if len(audio_buffer) == 0:
                    print("⚠️  오디오 데이터가 없습니다. 계속 시도 중...")
                    time.sleep(1)
                    continue

                # RMS 계산
                rms = np.sqrt(np.mean(audio_buffer**2))

                if rms < 0.001:
                    print(f"\n⚠️  소리가 너무 작습니다 (RMS: {rms:.6f}). 더 큰 소리로 말해주세요...")
                    time.sleep(1)
                    continue

                print(f"\n⏺️  [{time.strftime('%H:%M:%S')}] 분석 중...")

                # 양쪽 분석
                result_without = self.analyze_without_filtering(audio_buffer, self.audio_capture.sample_rate)
                result_with = self.analyze_with_filtering(audio_buffer, self.audio_capture.sample_rate)

                # 결과 출력
                self.print_side_by_side_results(result_without, result_with, rms)

                print("다음 분석까지 대기... (Ctrl+C로 중단)")

        except KeyboardInterrupt:
            pass
        finally:
            self.audio_capture.stop_recording()
            print("\n\n🛑 비교 모드 종료")


def main():
    parser = argparse.ArgumentParser(description="Real-time filtering comparison")
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Analysis interval in seconds (default: 3.0)')

    args = parser.parse_args()

    print("🚗 실시간 필터링 효과 비교")
    print("-" * 100)

    system = RealtimeFilteringComparison()
    system.run_continuous_comparison(duration=args.duration)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n프로그램이 종료되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
