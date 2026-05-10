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
    def __init__(self, debug_mode=False, comparison_mode=False, use_custom_model=True):
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
        # 우선순위: binary (Two-Class) > verified > v4 > v2 > v1
        binary_path = Path(__file__).parent / "data" / "models" / "car_classifier_binary.pkl"
        verified_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_verified.pkl"
        oneclass_v4_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v4.pkl"
        oneclass_v2_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v2.pkl"
        oneclass_model_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass.pkl"

        if binary_path.exists():
            model_path = str(binary_path)
            if use_custom_model:
                print("✅ Binary 모델 로드됨 (정상 vs 노킹, 98.1% 정확도)")
        elif verified_path.exists():
            model_path = str(verified_path)
            if use_custom_model:
                print("✅ Verified 모델 로드됨 (수동 검수 데이터)")
        elif oneclass_v4_path.exists():
            model_path = str(oneclass_v4_path)
        elif oneclass_v2_path.exists():
            model_path = str(oneclass_v2_path)
        elif oneclass_model_path.exists():
            model_path = str(oneclass_model_path)
        else:
            model_path = None

        self.model_path = model_path
        self.diagnoser = CarNoiseDiagnoser(model_path=model_path, use_custom_model=use_custom_model)

        # Display current model status
        if self.diagnoser.mode == "custom":
            print("📊 현재 모델: Custom 모델 (학습된 노킹 감지 모델)")
        else:
            print("📊 현재 모델: YAMNet Baseline (범용 오디오 분류 모델)")

        # Show comparison mode status
        if self.comparison_mode and self.diagnoser.mode == "custom":
            print("🔄 비교 모드 활성화: Baseline과 Custom 모델을 동시에 비교합니다.")
        elif self.comparison_mode:
            print("⚠️  비교 모드는 Custom 모델이 있을 때만 작동합니다.")
            self.comparison_mode = False

    def switch_model(self, use_custom: bool):
        """Switch between custom model and baseline YAMNet

        Args:
            use_custom: If True, switch to custom model. If False, switch to YAMNet baseline.
        """
        if use_custom and not self.model_path:
            print("❌ Custom 모델이 없습니다. Baseline 모드를 유지합니다.")
            return self.diagnoser.mode

        mode = self.diagnoser.switch_model(use_custom)

        if mode == "custom":
            print("✅ Custom 모델로 전환됨 (학습된 노킹 감지 모델)")
        else:
            print("✅ YAMNet Baseline으로 전환됨 (범용 오디오 분류 모델)")

        # Disable comparison mode if switching to baseline
        if mode == "baseline" and self.comparison_mode:
            self.comparison_mode = False
            print("⚠️  Baseline 모드에서는 비교 모드를 사용할 수 없습니다.")

        return mode

    def analyze_audio_file(self, file_path: str) -> dict:
        """파일에서 오디오를 로드하여 슬라이딩 윈도우로 분석"""
        print(f"오디오 파일 분석 중: {file_path}")

        # Load audio file
        audio_data, sample_rate = AudioFileLoader.load_audio(file_path)

        if len(audio_data) == 0:
            return {"error": "오디오 파일을 읽을 수 없습니다."}

        # Sliding window analysis (3초 윈도우, 1.5초 hop)
        window_size = 3.0
        hop_size = 1.5
        window_samples = int(window_size * sample_rate)
        hop_samples = int(hop_size * sample_rate)

        knocking_detected = False
        max_confidence = 0.0
        total_windows = 0
        knocking_windows = 0
        all_confidences = []  # Track all confidence scores

        print(f"슬라이딩 윈도우 분석 중 ({window_size}초 윈도우)...")

        for start_sample in range(0, len(audio_data) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            segment = audio_data[start_sample:end_sample]

            # Skip if too quiet
            rms = np.sqrt(np.mean(segment ** 2))
            if rms < 0.01:
                continue

            total_windows += 1
            time_sec = start_sample / sample_rate

            # Analyze this window
            result = self._analyze_audio_data(segment, sample_rate)

            # 비교 모드일 때는 각 윈도우마다 결과 출력
            if self.comparison_mode and result.get('success'):
                diagnosis = result.get('diagnosis', {})
                if diagnosis.get('mode') == 'comparison':
                    print(f"\n{'='*60}")
                    print(f"⏱️  구간: {time_sec:.1f}초 - {time_sec + window_size:.1f}초")
                    print(f"{'='*60}")
                    self._print_comparison_report_compact(result)

                    # 비교 모드에서는 custom 결과로 노킹 판단
                    custom_result = diagnosis.get('custom', {})
                    if custom_result.get('issues'):
                        knocking_detected = True
                        knocking_windows += 1
                        max_confidence = max(max_confidence, custom_result.get('confidence', 0))
                    continue

            if result.get('success'):
                confidence = result['diagnosis'].get('confidence', 0)
                all_confidences.append(confidence)

                if result['diagnosis'].get('issues'):
                    knocking_detected = True
                    knocking_windows += 1
                    max_confidence = max(max_confidence, confidence)
                    print(f"  🚨 {time_sec:.1f}초: 노킹 감지 (신뢰도: {confidence:.0%})")

        # Return summary result
        # 비교 모드일 때는 간단한 요약만 출력
        if self.comparison_mode:
            print(f"\n{'='*60}")
            print("📊 전체 요약")
            print(f"{'='*60}")
            print(f"분석 구간: {total_windows}개")
            if knocking_detected:
                print(f"🚨 노킹 감지 구간: {knocking_windows}개")
                print(f"최대 신뢰도: {max_confidence:.1%}")
            else:
                print("✅ 모든 구간 정상")
            print(f"{'='*60}\n")

            # 비교 모드는 print_diagnosis_report 호출 안 함
            return {'success': True, 'skip_print': True}

        # Calculate average confidence from all windows
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.5

        # 일반 모드 - audio_info 포함해서 반환
        if knocking_detected:
            return {
                'success': True,
                'audio_info': {
                    'duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate,
                    'rms_level': float(np.sqrt(np.mean(audio_data**2)))
                },
                'diagnosis': {
                    'mode': 'custom',
                    'overall_status': CarPartStatus.WARNING,
                    'issues': [{
                        'part': '엔진',
                        'status': 'WARNING',
                        'description': f'{knocking_windows}/{total_windows} 구간에서 노킹 감지',
                        'confidence': max_confidence
                    }],
                    'recommendations': ['정비소 방문을 권장합니다.'],
                    'confidence': max_confidence,
                    'detected_sounds': [],
                    'prediction': 1
                }
            }
        else:
            return {
                'success': True,
                'audio_info': {
                    'duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate,
                    'rms_level': float(np.sqrt(np.mean(audio_data**2)))
                },
                'diagnosis': {
                    'mode': 'custom',
                    'overall_status': CarPartStatus.NORMAL,
                    'issues': [],
                    'recommendations': ['정상 작동 중입니다.'],
                    'confidence': float(avg_confidence),  # Use actual average confidence
                    'detected_sounds': [],
                    'prediction': 0
                }
            }
    
    def analyze_realtime_continuous(self) -> None:
        """연속 실시간 오디오 분석"""
        print("\n" + "━"*60)
        print("🚗 자동차 노킹 감지 시스템 - 실시간 모니터링")
        print("━"*60)
        if self.comparison_mode:
            print("📊 비교 모드: YAMNet vs Custom 모델")
        else:
            print(f"📊 모델: {self.diagnoser.mode.upper()}")
        print("⏹  Ctrl+C를 눌러 중단")
        print("━"*60 + "\n")

        try:
            self.audio_capture.start_recording()

            analysis_interval = 3.0  # 3초마다 분석

            while True:
                try:
                    # 3초 동안의 오디오 수집
                    audio_buffer = self.audio_capture.get_audio_buffer(analysis_interval)

                    if len(audio_buffer) == 0:
                        time.sleep(1)
                        continue

                    # 슬라이딩 윈도우 분석 (1.5초 윈도우, 0.5초 hop)
                    sample_rate = self.audio_capture.sample_rate
                    window_size = 1.5
                    hop_size = 0.5  # 0.5초 hop으로 변경 → 3초에서 4개 윈도우 생성
                    window_samples = int(window_size * sample_rate)
                    hop_samples = int(hop_size * sample_rate)

                    # 전체 버퍼 RMS 체크
                    buffer_rms = np.sqrt(np.mean(audio_buffer ** 2))

                    # 디버그 정보 (디버그 모드일 때만)
                    if self.debug_mode:
                        print(f"  [DEBUG] 버퍼 RMS: {buffer_rms:.6f}, 길이: {len(audio_buffer)}")

                    knocking_detected = False
                    max_confidence = 0.0
                    window_count = 0
                    analyzed_count = 0
                    window_results = []  # 윈도우별 결과 저장

                    # 비교 모드가 아닐 때만 분석 중 메시지
                    timestamp = time.strftime('%H:%M:%S')
                    if not self.comparison_mode:
                        print(f"[{timestamp}] 분석 중...", end='\r')

                    for start_sample in range(0, len(audio_buffer) - window_samples, hop_samples):
                        end_sample = start_sample + window_samples
                        segment = audio_buffer[start_sample:end_sample]
                        window_count += 1

                        # Skip if too quiet
                        rms = np.sqrt(np.mean(segment ** 2))

                        if self.debug_mode:
                            print(f"  [DEBUG] Window {window_count}: RMS={rms:.6f}", end="")

                        if rms < 0.01:
                            if self.debug_mode:
                                print(" → SKIP (too quiet)")
                            continue

                        analyzed_count += 1
                        if self.debug_mode:
                            print(" → ANALYZING")

                        # Analyze this window
                        result = self._analyze_audio_data(segment, sample_rate)

                        if result.get('success'):
                            diagnosis = result['diagnosis']

                            # 비교 모드일 때
                            if self.comparison_mode and diagnosis.get('mode') == 'comparison':
                                baseline = diagnosis['baseline']
                                custom = diagnosis['custom']

                                # 노킹 위험도 점수 계산 (0-100점)
                                # YAMNet: "Engine knocking" 클래스만 사용
                                if baseline['issues']:
                                    # 노킹 관련 문제를 감지했을 때
                                    baseline_risk = int(baseline['confidence'] * 100)
                                else:
                                    # 정상일 때: YAMNet이 "Engine knocking" 클래스를 얼마나 감지했는지 확인
                                    engine_knocking_confidence = 0.0
                                    for sound in baseline.get('detected_sounds', []):
                                        sound_name_lower = sound['sound_type'].lower()
                                        # "Engine knocking" 문자열을 정확히 포함하는지 확인
                                        if 'engine knocking' in sound_name_lower:
                                            engine_knocking_confidence = max(engine_knocking_confidence, sound['confidence'])

                                    baseline_risk = int(engine_knocking_confidence * 100)

                                # Custom: probabilities에서 노킹 확률 가져오기
                                if custom.get('probabilities'):
                                    custom_risk = int(custom['probabilities'][1] * 100)  # 노킹 확률 (index 1)
                                else:
                                    custom_risk = int(custom['confidence'] * 100) if custom['issues'] else int((1 - custom['confidence']) * 100)

                                # Custom 결과로 노킹 판단
                                if custom['issues']:
                                    knocking_detected = True
                                    max_confidence = max(max_confidence, custom['confidence'])

                                # 결과 저장 (나중에 출력)
                                window_results.append({
                                    'baseline_risk': baseline_risk,
                                    'custom_risk': custom_risk,
                                    'baseline': baseline,
                                    'custom': custom
                                })
                            else:
                                # 일반 모드
                                # YAMNet 분류 먼저 보기 (디버그 모드일 때만)
                                if self.debug_mode:
                                    mediapipe_results = self.classifier.classify_audio(segment, sample_rate)
                                    if mediapipe_results:
                                        top_5 = self.classifier.get_top_predictions(mediapipe_results, top_k=5)
                                        print("    [YAMNET Top 5]")
                                        for i, pred in enumerate(top_5, 1):
                                            print(f"      {i}. {pred['category_name']:<30} {pred['score']:5.1%}")

                                pred = diagnosis.get('prediction', 'unknown')
                                conf = diagnosis.get('confidence', 0)
                                anomaly = diagnosis.get('anomaly_score', 'N/A')
                                has_issues = len(diagnosis.get('issues', [])) > 0

                                print(f"    → Pred={pred}, Conf={conf:.1%}, Anomaly={anomaly}, Issues={has_issues}")

                                if has_issues:
                                    knocking_detected = True
                                    max_confidence = max(max_confidence, conf)
                                    print(f"    → 🚨 KNOCKING DETECTED!")
                                    break  # 하나라도 감지되면 바로 경고

                    # 결과 출력
                    timestamp = time.strftime('%H:%M:%S')

                    if self.debug_mode:
                        print(f"  [DEBUG] 총 {window_count}개 윈도우, {analyzed_count}개 분석, 노킹={knocking_detected}")

                    # 1. 먼저 상태 메시지 출력
                    if knocking_detected:
                        print(f"\r🚨 [{timestamp}] 노킹 감지! (신뢰도: {max_confidence:.0%})" + " "*20)
                    else:
                        print(f"\r✅ [{timestamp}] 정상        " + " "*20)

                    # 2. 그 다음 비교 모드 박스들 출력
                    if self.comparison_mode and window_results:
                        # 위험도에 따른 상태 결정 함수
                        def get_status(risk_score):
                            if risk_score >= 50:
                                return "🚨 경고"
                            else:
                                return "✅ 정상"

                        for result in window_results:
                            baseline_risk = result['baseline_risk']
                            custom_risk = result['custom_risk']
                            baseline = result['baseline']
                            custom = result['custom']

                            baseline_status = get_status(baseline_risk)
                            custom_status = get_status(custom_risk)

                            # 박스 형태로 예쁘게 출력 (가로 배치)
                            if not self.debug_mode:
                                # 점수 포맷 (오른쪽 정렬, 3자리)
                                yamnet_score = f"{baseline_risk:3d}점"
                                custom_score = f"{custom_risk:3d}점"

                                # 한글/이모지는 터미널에서 2칸 차지
                                # 각 셀: 공백(1) + YAMNet:(8) + ✅ 정상(7) + 공백(1) + 점수(5) + 공백(1) = 23칸
                                yamnet_line = f"YAMNet: {baseline_status} {yamnet_score}"
                                custom_line = f"Custom: {custom_status} {custom_score}"

                                print(f"    ┌{'─'*23}┬{'─'*23}┐")
                                print(f"    │ {yamnet_line} │ {custom_line} │")
                                print(f"    └{'─'*23}┴{'─'*23}┘")
                            else:
                                # 디버그 모드는 상세 정보
                                custom_pred = custom.get('prediction', 'unknown')
                                print(f"    YAMNet: {baseline_status} ({baseline['confidence']:.1%})")
                                print(f"    Custom: {custom_status} (예측={custom_pred}, 신뢰도={custom['confidence']:.1%})")

                        # 기준 설명 (마지막에 한 번만)
                        if not self.debug_mode:
                            print(f"                [기준: 50점 이상 경고]")
                        print()  # 각 진단 사이 공백

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
            print("\n" + "━"*60)
            print("🛑 모니터링 종료")
            print("━"*60)
    
    def analyze_realtime_single(self, duration: float = 5.0) -> dict:
        """단일 실시간 오디오 분석 (슬라이딩 윈도우)"""
        print(f"🎙️  {duration}초간 오디오 수집 중...")

        try:
            self.audio_capture.start_recording()
            audio_buffer = self.audio_capture.get_audio_buffer(duration)
            self.audio_capture.stop_recording()

            if len(audio_buffer) == 0:
                return {"error": "오디오 데이터를 받을 수 없습니다."}

            sample_rate = self.audio_capture.sample_rate

            # Sliding window analysis (3초 윈도우, 1.5초 hop)
            window_size = 3.0
            hop_size = 1.5
            window_samples = int(window_size * sample_rate)
            hop_samples = int(hop_size * sample_rate)

            knocking_detected = False
            max_confidence = 0.0
            total_windows = 0
            knocking_windows = 0
            all_confidences = []  # Track all confidence scores

            for start_sample in range(0, len(audio_buffer) - window_samples, hop_samples):
                end_sample = start_sample + window_samples
                segment = audio_buffer[start_sample:end_sample]

                # Skip if too quiet
                rms = np.sqrt(np.mean(segment ** 2))
                if rms < 0.01:
                    continue

                total_windows += 1

                # Analyze this window
                result = self._analyze_audio_data(segment, sample_rate)

                if result.get('success'):
                    confidence = result['diagnosis'].get('confidence', 0)
                    all_confidences.append(confidence)

                    if result['diagnosis'].get('issues'):
                        knocking_detected = True
                        knocking_windows += 1
                        max_confidence = max(max_confidence, confidence)

            # Calculate average confidence from all windows
            avg_confidence = np.mean(all_confidences) if all_confidences else 0.5

            # Return summary result
            if knocking_detected:
                return {
                    'success': True,
                    'diagnosis': {
                        'mode': 'custom',
                        'overall_status': CarPartStatus.WARNING,
                        'issues': [{
                            'part': '엔진',
                            'status': 'WARNING',
                            'description': f'{knocking_windows}/{total_windows} 구간에서 노킹 감지',
                            'confidence': max_confidence
                        }],
                        'recommendations': ['정비소 방문을 권장합니다.'],
                        'confidence': max_confidence,
                        'detected_sounds': [],
                        'prediction': 1
                    }
                }
            else:
                return {
                    'success': True,
                    'diagnosis': {
                        'mode': 'custom',
                        'overall_status': CarPartStatus.NORMAL,
                        'issues': [],
                        'recommendations': ['정상 작동 중입니다.'],
                        'confidence': float(avg_confidence),  # Use actual average confidence
                        'detected_sounds': [],
                        'prediction': 0
                    }
                }

        except Exception as e:
            return {"error": f"오디오 캡처 오류: {str(e)}"}
    
    def _analyze_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> dict:
        """오디오 데이터 분석"""
        try:
            # 0. 오디오 정규화 (볼륨 통일)
            current_rms = np.sqrt(np.mean(audio_data ** 2))
            if current_rms > 0.001:  # 너무 작은 소리가 아니면
                target_rms = 0.1  # 목표 RMS
                audio_data = audio_data * (target_rms / current_rms)
                if self.debug_mode:
                    print(f"  [NORM] RMS {current_rms:.6f} → {target_rms:.6f} (증폭: {target_rms/current_rms:.1f}x)")

            # 1. YAMNet 분류 (조용히 실행)
            mediapipe_results = self.classifier.classify_audio(audio_data, sample_rate)

            # 2. Embedding 추출 (Custom classifier용)
            embedding = None
            if self.diagnoser.mode == "custom":
                embedding = self.classifier.extract_embedding(audio_data, sample_rate)

            # 3. 진단 수행

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

        # One-Class 모델의 경우 간결한 출력
        is_one_class = hasattr(self.diagnoser, 'is_one_class') and self.diagnoser.is_one_class

        if is_one_class and not self.debug_mode:
            self._print_oneclass_simple_report(result)
            return

        # 일반 모드 또는 디버그 모드 상세 출력
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

        # YAMNet 전체 분류 결과 (상위 10개) - 디버그 모드에서만
        if self.debug_mode and result.get('mediapipe_results'):
            all_predictions = self.classifier.get_top_predictions(result['mediapipe_results'], top_k=10)
            if all_predictions:
                print(f"\n🤖 YAMNet 분류 결과 (상위 10개):")
                for i, pred in enumerate(all_predictions, 1):
                    print(f"   {i:2}. {pred['category_name']:<30} ({pred['score']:.1%})")

        # 감지된 소리 - 간략하게
        if diagnosis['detected_sounds'] and self.debug_mode:
            print(f"\n🔊 차량 관련 소리 감지 결과:")
            for sound in diagnosis['detected_sounds'][:5]:  # Top 5
                print(f"   - {sound['part']}: {sound['sound_type']} ({sound['confidence']:.1%})")

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

    def _print_oneclass_simple_report(self, result: dict):
        """One-Class 모델용 간결한 진단 결과 출력"""
        diagnosis = result['diagnosis']

        print("\n" + "="*50)
        print("🔍 엔진 노킹 진단 결과")
        print("="*50)

        # 노킹 감지 여부에 따른 메시지
        if diagnosis['issues']:
            # 노킹 감지!
            confidence = diagnosis['confidence']
            print(f"\n🚨 엔진 노킹 감지됨! (신뢰도: {confidence:.0%})")

            # 상세 설명
            for issue in diagnosis['issues'][:1]:  # 첫 번째만
                if issue.get('description'):
                    print(f"   └─ {issue['description']}")

            # 조치 사항
            if diagnosis['recommendations']:
                print(f"\n💡 조치 사항:")
                for rec in diagnosis['recommendations'][:2]:  # 최대 2개
                    print(f"   • {rec}")

            # Anomaly score (디버그용)
            if self.debug_mode and 'anomaly_score' in diagnosis:
                score = diagnosis['anomaly_score']
                print(f"\n📈 이상치 점수: {score:.3f}")
        else:
            # 정상
            confidence = diagnosis['confidence']
            print(f"\n✅ 정상 - 엔진 노킹 없음 (신뢰도: {confidence:.0%})")

            # 디버그 정보
            if self.debug_mode and 'anomaly_score' in diagnosis:
                score = diagnosis['anomaly_score']
                print(f"   └─ 이상치 점수: {score:.3f}")

        print("\n" + "="*50)

    def _print_continuous_report(self, result: dict):
        """연속 모드용 간략한 진단 결과 출력 (노킹 진단만)"""
        if not result.get('success', False):
            return  # 오류도 조용히 무시

        diagnosis = result['diagnosis']
        timestamp = time.strftime('%H:%M:%S')

        # One-Class 모델의 경우 더 간결하게
        is_one_class = hasattr(self.diagnoser, 'is_one_class') and self.diagnoser.is_one_class

        if is_one_class:
            if diagnosis['issues']:
                # 노킹 감지만 표시
                confidence = diagnosis['confidence']
                print(f"🚨 [{timestamp}] 노킹 감지 (신뢰도: {confidence:.0%})")
            else:
                # 정상은 아주 간결하게
                print(f"✅ [{timestamp}] 정상")
        else:
            # 일반 모델도 간결하게
            if diagnosis['issues']:
                print(f"⚠️ [{timestamp}] 문제 감지")
            else:
                print(f"✅ [{timestamp}] 정상")

    def _print_comparison_report_compact(self, result: dict):
        """비교 모드 간소화 리포트 (윈도우별 출력용)"""
        diagnosis = result['diagnosis']
        baseline = diagnosis['baseline']
        custom = diagnosis['custom']

        # Baseline 결과
        baseline_status = "🚨 노킹" if baseline['issues'] else "✅ 정상"
        baseline_conf = baseline['confidence']

        # Custom 결과
        custom_status = "🚨 노킹" if custom['issues'] else "✅ 정상"
        custom_conf = custom['confidence']

        # 비교 출력
        print(f"YAMNet Baseline:  {baseline_status} (신뢰도: {baseline_conf:.1%})")
        print(f"Binary Custom:    {custom_status} (신뢰도: {custom_conf:.1%})")

        # 일치 여부
        agree = (len(baseline['issues']) > 0) == (len(custom['issues']) > 0)
        if agree:
            print("✅ 두 모델 일치")
        else:
            print("⚠️  두 모델 불일치")

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
    print("🚗 엔진 노킹 진단 시스템")
    print("AI 기반 실시간 소음 분석")
    print("-" * 40)

    # 기본 설정 (모든 선택 제거)
    debug_mode = False  # 디버그 모드 OFF
    use_custom_model = True  # Custom 모델 사용

    # Custom 모델 존재 여부 확인
    binary_path = Path(__file__).parent / "data" / "models" / "car_classifier_binary.pkl"
    verified_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_verified.pkl"
    oneclass_v4_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v4.pkl"
    oneclass_v2_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v2.pkl"
    oneclass_model_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass.pkl"

    has_custom_model = any([
        binary_path.exists(),
        verified_path.exists(),
        oneclass_v4_path.exists(),
        oneclass_v2_path.exists(),
        oneclass_model_path.exists()
    ])

    # 비교 모드: Custom 모델이 있으면 자동으로 ON
    comparison_mode = has_custom_model

    # 시스템 초기화
    system = CarNoiseDiagnosisSystem(
        debug_mode=debug_mode,
        comparison_mode=comparison_mode,
        use_custom_model=use_custom_model
    )

    print("\n" + "-" * 40)

    # 바로 연속 모니터링 시작
    try:
        system.analyze_realtime_continuous()
    except KeyboardInterrupt:
        pass

    # 종료 메시지는 analyze_realtime_continuous 내부에서 출력됨


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {e}")
        print("requirements.txt의 패키지들이 모두 설치되었는지 확인해보세요.")