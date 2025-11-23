#!/usr/bin/env python3
"""
실시간 테스트 - 다양한 시나리오로 혼합 음향 환경 테스트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from main import CarNoiseDiagnosisSystem
import numpy as np
import time

def test_mixed_audio_scenarios():
    """다양한 혼합 음향 시나리오 테스트"""
    print("🧪 혼합 음향 환경 테스트")
    print("=" * 50)
    
    system = CarNoiseDiagnosisSystem()
    
    scenarios = [
        ("정상 엔진 + 조용한 환경", create_normal_engine_sound),
        ("문제 있는 엔진 + 사람 말소리", create_engine_with_voice),
        ("브레이크 문제 + 배경 소음", create_brake_issue_with_noise),
        ("베어링 문제 + 라디오 소리", create_bearing_with_radio),
    ]
    
    for scenario_name, generator_func in scenarios:
        print(f"\n🎵 시나리오: {scenario_name}")
        print("-" * 30)
        
        # 시나리오별 오디오 생성
        fake_audio, sample_rate = generator_func()
        
        # 분석 수행
        result = system._analyze_audio_data(fake_audio, sample_rate)
        
        # 결과 출력
        system.print_diagnosis_report(result)
        
        print("\n" + "="*30)
        time.sleep(1)  # 잠시 대기

def create_normal_engine_sound():
    """정상 엔진 소리 시뮬레이션"""
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 정상적인 엔진 소리 (주로 저주파)
    engine_base = 0.4 * np.sin(2 * np.pi * 85 * t)  # 85Hz 기본 주파수
    engine_2nd = 0.2 * np.sin(2 * np.pi * 170 * t)  # 2차 하모닉
    engine_3rd = 0.1 * np.sin(2 * np.pi * 255 * t)  # 3차 하모닉
    background_noise = 0.05 * np.random.normal(0, 1, len(t))
    
    audio = engine_base + engine_2nd + engine_3rd + background_noise
    return audio.astype(np.float32), sample_rate

def create_engine_with_voice():
    """엔진 문제 + 사람 말소리"""
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 문제 있는 엔진 (불규칙하고 고주파 성분 포함)
    engine_base = 0.3 * np.sin(2 * np.pi * 75 * t)  # 약간 낮은 기본 주파수
    engine_irregular = 0.2 * np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))  # 불규칙성
    problem_noise = 0.25 * np.sin(2 * np.pi * 2500 * t) * np.exp(-t/2)  # 고주파 문제음
    
    # 사람 목소리 시뮬레이션 (300-3400 Hz 대역)
    voice_freq1 = 0.15 * np.sin(2 * np.pi * 500 * t) * np.sin(2 * np.pi * 5 * t)  # 기본 음성
    voice_freq2 = 0.1 * np.sin(2 * np.pi * 1200 * t) * np.sin(2 * np.pi * 3 * t)  # 고주파 음성 성분
    voice_modulation = 1 + 0.5 * np.sin(2 * np.pi * 4 * t)  # 음성 변조
    voice = (voice_freq1 + voice_freq2) * voice_modulation
    
    background = 0.05 * np.random.normal(0, 1, len(t))
    
    audio = engine_base + engine_irregular + problem_noise + voice + background
    return audio.astype(np.float32), sample_rate

def create_brake_issue_with_noise():
    """브레이크 문제 + 배경 소음"""
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 브레이크 관련 고주파 소음 (삐걱거리는 소리)
    brake_squeal = 0.4 * np.sin(2 * np.pi * 2800 * t) * np.exp(-t/3)
    brake_grind = 0.3 * np.sin(2 * np.pi * 1800 * t) * (1 + 0.5 * np.sin(2 * np.pi * 8 * t))
    
    # 일반적인 도로 소음
    road_noise = 0.2 * np.random.normal(0, 1, len(t))
    wind_noise = 0.1 * np.sin(2 * np.pi * 150 * t) * np.sin(2 * np.pi * 0.5 * t)
    
    # 기본 엔진음
    engine = 0.2 * np.sin(2 * np.pi * 90 * t)
    
    audio = brake_squeal + brake_grind + road_noise + wind_noise + engine
    return audio.astype(np.float32), sample_rate

def create_bearing_with_radio():
    """베어링 문제 + 라디오 소리"""
    duration = 3.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 베어링 문제 (연속적인 고주파 소음)
    bearing_noise = 0.3 * np.sin(2 * np.pi * 3200 * t)
    bearing_variation = 0.2 * np.sin(2 * np.pi * 3800 * t) * (1 + 0.3 * np.sin(2 * np.pi * 1 * t))
    
    # 라디오/음악 시뮬레이션 (복잡한 주파수 스펙트럼)
    radio_bass = 0.15 * np.sin(2 * np.pi * 80 * t) * np.sin(2 * np.pi * 2 * t)
    radio_mid = 0.1 * np.sin(2 * np.pi * 800 * t) * np.sin(2 * np.pi * 3 * t)
    radio_high = 0.08 * np.sin(2 * np.pi * 4000 * t) * np.sin(2 * np.pi * 1.5 * t)
    
    # 기본 엔진음
    engine = 0.25 * np.sin(2 * np.pi * 88 * t)
    
    background = 0.03 * np.random.normal(0, 1, len(t))
    
    audio = bearing_noise + bearing_variation + radio_bass + radio_mid + radio_high + engine + background
    return audio.astype(np.float32), sample_rate

def test_voice_detection():
    """음성 감지 기능 테스트"""
    print("\n🎤 음성 감지 기능 테스트")
    print("=" * 30)
    
    system = CarNoiseDiagnosisSystem()
    
    test_cases = [
        ("순수 엔진음", create_normal_engine_sound),
        ("엔진음 + 목소리", create_engine_with_voice),
    ]
    
    for case_name, generator in test_cases:
        print(f"\n테스트: {case_name}")
        audio, sr = generator()
        
        voice_analysis = system.preprocessor.detect_voice_activity(audio, sr)
        print(f"음성 감지: {'예' if voice_analysis['voice_detected'] else '아니오'}")
        print(f"음성 비율: {voice_analysis['voice_ratio']:.1%}")
        print(f"기계음 비율: {voice_analysis['mechanical_ratio']:.1%}")
        print(f"음향 타입: {voice_analysis['audio_type']}")

if __name__ == "__main__":
    print("🚗 실시간 혼합 음향 테스트")
    print("이 테스트는 다양한 시나리오에서 시스템이 어떻게 반응하는지 확인합니다.")
    print("실제 마이크 입력 대신 시뮬레이션된 오디오를 사용합니다.\n")
    
    try:
        test_voice_detection()
        test_mixed_audio_scenarios()
        
        print("\n" + "="*50)
        print("✅ 모든 시나리오 테스트 완료!")
        print("\n💡 실제 테스트 방법:")
        print("1. ffmpeg 설치: brew install ffmpeg")
        print("2. python main.py 실행")
        print("3. '1. 실시간 오디오 분석' 선택")
        print("4. 마이크에 엔진음 + 말소리 동시에 들려주기")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        print("requirements.txt의 패키지들이 설치되어 있는지 확인해보세요.")