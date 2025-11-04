#!/usr/bin/env python3
"""
Car Noise Diagnosis System
실시간 오디오 입력을 받아 자동차 소음을 분석하고 고장 진단을 수행합니다.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from audio.capture import AudioCapture, AudioFileLoader
from models.mediapipe_classifier import MediaPipeAudioClassifier, AudioPreprocessor
from diagnosis.analyzer import CarNoiseDiagnoser, CarPartStatus


class CarNoiseDiagnosisSystem:
    def __init__(self, debug_mode=False, comparison_mode=False):
        self.audio_capture = AudioCapture()
        self.debug_mode = debug_mode
        self.comparison_mode = comparison_mode

        # YAMNet 모델 경로 설정
        yamnet_model_path = Path(__file__).parent / "data" / "models" / "yamnet.tflite"
        if not yamnet_model_path.exists():
            print(f"⚠️  모델 파일이 없습니다: {yamnet_model_path}")
            print("YAMNet 모델을 다운로드하려면 다음 명령어를 실행하세요:")
            print("curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite")

        self.classifier = MediaPipeAudioClassifier(
            model_path=str(yamnet_model_path),
            max_results=10,
            score_threshold=0.0  # 모든 결과 보기
        )
        self.preprocessor = AudioPreprocessor()

        # Custom classifier 경로 (학습된 모델)
        custom_model_path = Path(__file__).parent / "data" / "models" / "car_classifier.pkl"
        self.diagnoser = CarNoiseDiagnoser(
            model_path=str(custom_model_path) if custom_model_path.exists() else None
        )

        # Show comparison mode status
        if self.comparison_mode and self.diagnoser.mode == "custom":
            print("🔄 비교 모드 활성화: Baseline과 Custom 모델을 동시에 비교합니다.")
        elif self.comparison_mode:
            print("⚠️  비교 모드는 Custom 모델이 있을 때만 작동합니다.")
            self.comparison_mode = False
        
    def analyze_audio_file(self, file_path: str) -> dict:
        """파일에서 오디오를 로드하여 분석"""
        print(f"오디오 파일 분석 중: {file_path}")
        
        # Load audio file
        audio_data, sample_rate = AudioFileLoader.load_audio(file_path)
        
        if len(audio_data) == 0:
            return {"error": "오디오 파일을 읽을 수 없습니다."}
            
        return self._analyze_audio_data(audio_data, sample_rate)
    
    def analyze_realtime_continuous(self) -> None:
        """연속 실시간 오디오 분석"""
        print("🎙️  연속 실시간 분석 모드")
        print("Ctrl+C를 눌러 중단할 수 있습니다...")
        print("마이크에 소리를 들려주세요...")
        print("-" * 50)
        
        try:
            self.audio_capture.start_recording()
            
            analysis_interval = 3.0  # 3초마다 분석
            
            while True:
                try:
                    # 3초 동안의 오디오 수집
                    audio_buffer = self.audio_capture.get_audio_buffer(analysis_interval)
                    
                    if len(audio_buffer) == 0:
                        print("⚠️  오디오 데이터가 없습니다. 계속 시도 중...")
                        time.sleep(1)
                        continue
                    
                    # 오디오 입력 상태 표시
                    rms = np.sqrt(np.mean(audio_buffer**2))
                    max_val = np.max(np.abs(audio_buffer))
                    print(f"🎙️  마이크 입력: RMS={rms:.4f}, 최대값={max_val:.4f}, 길이={len(audio_buffer)}")
                    
                    # 분석 수행
                    print(f"\n🔍 [{time.strftime('%H:%M:%S')}] 분석 중...")
                    result = self._analyze_audio_data(audio_buffer, self.audio_capture.sample_rate)
                    
                    if result.get('success', False):
                        # 간략한 결과만 표시 (연속 모드용)
                        self._print_continuous_report(result)
                    else:
                        print(f"❌ 분석 오류: {result.get('error', '알 수 없음')}")
                    
                    print("-" * 50)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ 분석 중 오류: {e}")
                    time.sleep(1)
                    continue
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.audio_capture.stop_recording()
            print("\n🛑 연속 분석을 중단했습니다.")
    
    def analyze_realtime_single(self, duration: float = 5.0) -> dict:
        """단일 실시간 오디오 분석 (기존 방식)"""
        print(f"실시간 오디오 분석 시작 ({duration}초)")
        print("마이크에 소리를 들려주세요...")
        
        try:
            self.audio_capture.start_recording()
            audio_buffer = self.audio_capture.get_audio_buffer(duration)
            self.audio_capture.stop_recording()
            
            if len(audio_buffer) == 0:
                return {"error": "오디오 데이터를 받을 수 없습니다."}
                
            return self._analyze_audio_data(audio_buffer, self.audio_capture.sample_rate)
            
        except Exception as e:
            return {"error": f"오디오 캡처 오류: {str(e)}"}
    
    def _analyze_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """오디오 데이터 분석"""
        try:
            # 1. YAMNet 분류 (직접 분류, 필터링 없음 - 실험적 검증 결과)
            print("🤖 YAMNet 분류 중...")
            mediapipe_results = self.classifier.classify_audio(audio_data, sample_rate)

            # YAMNet 분류 결과 즉시 표시
            if mediapipe_results:
                top_10 = self.classifier.get_top_predictions(mediapipe_results, top_k=10)
                print("   🤖 YAMNet 실시간 분류 결과 (Top 10):")
                for i, pred in enumerate(top_10, 1):
                    print(f"     {i:2}. {pred['category_name']:<25} {pred['score']:.1%}")

                # 차량 관련만 별도 표시
                vehicle_sounds = self.classifier.filter_vehicle_sounds(mediapipe_results)
                if vehicle_sounds:
                    print("   🚗 차량 관련 소리:")
                    for sound in vehicle_sounds[:3]:
                        print(f"     - {sound['category_name']}: {sound['score']:.1%}")
                else:
                    print("   ❌ 차량 관련 소리 감지되지 않음")

            # 2. Embedding 추출 (Custom classifier용)
            # 주의: extract_embedding() 내부에서 오디오 특성도 함께 추출됨
            embedding = None
            if self.diagnoser.mode == "custom":
                print("특성 벡터 추출 중...")
                embedding = self.classifier.extract_embedding(audio_data, sample_rate)

            # 3. 진단 수행
            if self.comparison_mode:
                print(f"자동차 소음 진단 중... (비교 모드: Baseline + Custom)")
            else:
                print(f"자동차 소음 진단 중... (모드: {self.diagnoser.mode})")

            diagnosis = self.diagnoser.diagnose(
                {},  # audio_features는 사용되지 않음 (embedding에 포함됨)
                mediapipe_results,
                embedding=embedding,
                comparison_mode=self.comparison_mode
            )

            # 5. 결과 정리
            rms_level = float(np.sqrt(np.mean(audio_data**2)))

            result = {
                'timestamp': time.time(),
                'audio_info': {
                    'duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate,
                    'rms_level': rms_level
                },
                'mediapipe_results': mediapipe_results,
                'diagnosis': diagnosis,
                'success': True
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f"분석 중 오류 발생: {str(e)}",
                'success': False
            }
    
    def print_diagnosis_report(self, result: dict):
        """진단 결과를 보기 좋게 출력"""
        if not result.get('success', False):
            print(f"❌ 오류: {result.get('error', '알 수 없는 오류')}")
            return

        diagnosis = result['diagnosis']

        # Check if comparison mode
        if diagnosis.get('mode') == 'comparison':
            self._print_comparison_report(result)
            return

        print("\n" + "="*60)
        print("🚗 자동차 소음 진단 결과")
        print("="*60)

        audio_info = result['audio_info']
        
        # 기본 정보
        print(f"📊 분석 정보:")
        print(f"   - 분석 시간: {audio_info['duration']:.1f}초")
        print(f"   - 샘플링 레이트: {audio_info['sample_rate']} Hz")
        print(f"   - 음량 레벨: {audio_info['rms_level']:.3f}")
        print(f"   - 진단 신뢰도: {diagnosis['confidence']:.1%}")

        # 전체 상태
        status = diagnosis['overall_status']
        status_emoji = {
            CarPartStatus.NORMAL: "✅",
            CarPartStatus.WARNING: "⚠️",
            CarPartStatus.CRITICAL: "🚨"
        }
        
        print(f"\n🎯 전체 상태: {status_emoji.get(status, '❓')} {status.value}")
        
        # YAMNet 전체 분류 결과 (상위 10개)
        if result.get('mediapipe_results'):
            all_predictions = self.classifier.get_top_predictions(result['mediapipe_results'], top_k=10)
            if all_predictions:
                print(f"\n🤖 YAMNet 분류 결과 (상위 10개):")
                for i, pred in enumerate(all_predictions, 1):
                    print(f"   {i:2}. {pred['category_name']:<30} ({pred['score']:.1%})")
        
        # 감지된 소리
        if diagnosis['detected_sounds']:
            print(f"\n🔊 차량 관련 소리 감지 결과:")
            for sound in diagnosis['detected_sounds'][:5]:  # Top 5
                print(f"   - {sound['part']}: {sound['sound_type']} ({sound['confidence']:.1%})")
        else:
            print(f"\n🔊 차량 관련 소리: 감지되지 않음")
        
        # 발견된 문제들
        if diagnosis['issues']:
            print(f"\n🔍 발견된 문제들:")
            for i, issue in enumerate(diagnosis['issues'], 1):
                status_emoji_local = "🚨" if issue['status'] == "위험" else "⚠️"
                print(f"   {i}. {status_emoji_local} [{issue['part']}] {issue['description']}")
                print(f"      신뢰도: {issue['confidence']:.1%}")
        else:
            print(f"\n✅ 특별한 문제가 감지되지 않았습니다.")
        
        # 추천 사항
        if diagnosis['recommendations']:
            print(f"\n💡 추천 사항:")
            for i, rec in enumerate(diagnosis['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)
    
    def _print_continuous_report(self, result: dict):
        """연속 모드용 간략한 진단 결과 출력"""
        if not result.get('success', False):
            print(f"❌ 오류: {result.get('error', '알 수 없는 오류')}")
            return
            
        diagnosis = result['diagnosis']
        audio_info = result['audio_info']
        
        # 상태 이모지
        status = diagnosis['overall_status']
        status_emoji = {
            CarPartStatus.NORMAL: "✅",
            CarPartStatus.WARNING: "⚠️",
            CarPartStatus.CRITICAL: "🚨"
        }
        
        # 한 줄 요약
        print(f"상태: {status_emoji.get(status, '❓')} {status.value} | "
              f"신뢰도: {diagnosis['confidence']:.0%} | "
              f"음량: {audio_info['rms_level']:.3f}")
        
        # YAMNet 분류 결과 (항상 표시)
        if result.get('mediapipe_results'):
            top_5 = self.classifier.get_top_predictions(result['mediapipe_results'], top_k=5)
            if top_5:
                print("🤖 YAMNet Top 5:")
                for i, p in enumerate(top_5, 1):
                    print(f"  {i}. {p['category_name']:<20} {p['score']:.0%}")
            
            # 차량 관련 소리만 별도
            vehicle_sounds = self.classifier.filter_vehicle_sounds(result['mediapipe_results'])
            if vehicle_sounds:
                vehicle_list = [f"{s['category_name']}({s['score']:.0%})" for s in vehicle_sounds[:3]]
                print(f"🚗 차량음: {', '.join(vehicle_list)}")
        else:
            print("❌ YAMNet 분류 결과 없음")
        
        # 문제 발견 시만 상세 표시
        if diagnosis['issues']:
            critical_issues = [i for i in diagnosis['issues'] if i['status'] == CarPartStatus.CRITICAL.value]
            warning_issues = [i for i in diagnosis['issues'] if i['status'] == CarPartStatus.WARNING.value]
            
            if critical_issues:
                print("🚨 긴급:")
                for issue in critical_issues[:2]:  # 최대 2개만
                    print(f"  - {issue['part']}: {issue['description'][:40]}...")
            elif warning_issues:
                print("⚠️  주의:")
                for issue in warning_issues[:1]:  # 최대 1개만
                    print(f"  - {issue['part']}: {issue['description'][:40]}...")
        else:
            print("✅ 이상 없음")

    def _print_comparison_report(self, result: dict):
        """비교 모드 진단 결과 출력 (Baseline vs Custom)"""
        diagnosis = result['diagnosis']
        audio_info = result['audio_info']
        baseline = diagnosis['baseline']
        custom = diagnosis['custom']
        metrics = diagnosis['comparison_metrics']

        print("\n" + "="*70)
        print("🚗 자동차 소음 진단 결과 (비교 모드)")
        print("="*70)

        # 기본 정보
        print(f"📊 분석 정보:")
        print(f"   - 분석 시간: {audio_info['duration']:.1f}초")
        print(f"   - 샘플링 레이트: {audio_info['sample_rate']} Hz")
        print(f"   - 음량 레벨: {audio_info['rms_level']:.3f}")

        print("\n" + "-"*70)

        # Baseline 결과
        print("📌 YAMNet Baseline (범용 모델)")
        print("-"*70)

        status_emoji = {
            CarPartStatus.NORMAL: "✅",
            CarPartStatus.WARNING: "⚠️",
            CarPartStatus.CRITICAL: "🚨"
        }

        baseline_status = baseline['overall_status']
        print(f"   상태: {status_emoji.get(baseline_status, '❓')} {baseline_status.value}")
        print(f"   신뢰도: {baseline['confidence']:.1%}")

        if baseline['issues']:
            print(f"   문제:")
            for issue in baseline['issues'][:2]:
                print(f"     - [{issue['part']}] {issue['description']}")
        else:
            print(f"   문제: 감지되지 않음")

        print("\n" + "-"*70)

        # Custom 결과
        print("🎯 Custom Model (자동차 특화 학습 모델)")
        print("-"*70)

        custom_status = custom['overall_status']
        print(f"   상태: {status_emoji.get(custom_status, '❓')} {custom_status.value}")
        print(f"   신뢰도: {custom['confidence']:.1%}")

        if custom['issues']:
            print(f"   문제:")
            for issue in custom['issues'][:2]:
                print(f"     - [{issue['part']}] {issue['description']}")
        else:
            print(f"   문제: 감지되지 않음")

        print("\n" + "-"*70)

        # 비교 분석
        print("📈 비교 분석")
        print("-"*70)

        conf_diff = metrics['confidence_improvement']
        if conf_diff > 0:
            print(f"   ✅ Custom 모델이 {conf_diff:.1%}p 더 확신합니다")
        elif conf_diff < 0:
            print(f"   ⚠️  Baseline이 {abs(conf_diff):.1%}p 더 확신합니다")
        else:
            print(f"   ➡️  두 모델의 신뢰도가 동일합니다")

        if metrics['predictions_agree']:
            print(f"   ✅ 두 모델의 진단이 일치합니다")
        else:
            print(f"   ⚠️  두 모델의 진단이 다릅니다")

        # 추천
        print(f"\n💡 추천:")
        if custom['confidence'] > baseline['confidence'] + 0.1:
            print(f"   → Custom 모델 결과를 우선적으로 참고하세요 (신뢰도 높음)")
        elif baseline['confidence'] > custom['confidence'] + 0.1:
            print(f"   → Baseline 결과를 참고하세요 (Custom 모델 불확실)")
        else:
            print(f"   → 두 모델 모두 참고하세요 (신뢰도 유사)")

        if custom['recommendations']:
            print(f"\n📋 조치 사항:")
            for rec in custom['recommendations'][:3]:
                print(f"   - {rec}")

        print("\n" + "="*70)


def main():
    print("🚗 자동차 소음 진단 시스템 v2.0")
    print("YAMNet 기반 지능형 진단")
    print("-" * 40)

    # 디버그 모드 선택
    try:
        debug_choice = input("디버그 모드 (YAMNet 분류 상세 보기)? (y/N): ").lower().strip().replace('\r', '')
        debug_mode = debug_choice in ['y', 'yes']
    except (KeyboardInterrupt, EOFError):
        debug_mode = False

    # 비교 모드 선택 (Custom 모델 있을 때만)
    comparison_mode = False
    custom_model_path = Path(__file__).parent / "data" / "models" / "car_classifier.pkl"
    if custom_model_path.exists():
        try:
            comp_choice = input("비교 모드 (Baseline vs Custom 동시 비교)? (y/N): ").lower().strip().replace('\r', '')
            comparison_mode = comp_choice in ['y', 'yes']
        except (KeyboardInterrupt, EOFError):
            comparison_mode = False

    system = CarNoiseDiagnosisSystem(debug_mode=debug_mode, comparison_mode=comparison_mode)

    while True:
        print("\n선택하세요:")
        print("1. 연속 실시간 분석 (Ctrl+C로 중단)")
        print("2. 단발 실시간 분석 (5초)")
        print("3. 오디오 파일 분석")
        if system.diagnoser.mode == "custom" and not comparison_mode:
            print("4. 비교 모드 ON/OFF (현재: OFF)")
        elif system.diagnoser.mode == "custom" and comparison_mode:
            print("4. 비교 모드 ON/OFF (현재: ON)")
        print("5. 종료")

        choice = input("\n입력 (1-5): ").strip().replace('\r', '')
        
        if choice == '1':
            print("\n" + "-" * 40)
            system.analyze_realtime_continuous()
            
        elif choice == '2':
            print("\n" + "-" * 40)
            result = system.analyze_realtime_single(duration=5.0)
            system.print_diagnosis_report(result)
            
        elif choice == '3':
            file_path = input("오디오 파일 경로를 입력하세요: ").strip().replace('\r', '')
            if Path(file_path).exists():
                print("\n" + "-" * 40)
                result = system.analyze_audio_file(file_path)
                system.print_diagnosis_report(result)
            else:
                print("❌ 파일을 찾을 수 없습니다.")

        elif choice == '4':
            # Toggle comparison mode
            if system.diagnoser.mode == "custom":
                system.comparison_mode = not system.comparison_mode
                status = "ON" if system.comparison_mode else "OFF"
                print(f"✅ 비교 모드: {status}")
            else:
                print("❌ Custom 모델이 없어 비교 모드를 사용할 수 없습니다.")

        elif choice == '5':
            print("프로그램을 종료합니다.")
            break

        else:
            print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        print("requirements.txt의 패키지들이 모두 설치되었는지 확인해보세요.")