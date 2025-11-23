#!/usr/bin/env python3
"""
간단한 테스트 스크립트
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from main import CarNoiseDiagnosisSystem
import numpy as np

def test_basic_functionality():
    """기본 기능 테스트"""
    print("🧪 기본 기능 테스트")
    
    system = CarNoiseDiagnosisSystem()
    
    # 가짜 오디오 데이터로 테스트 (엔진 소리 시뮬레이션)
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 엔진 소리 시뮬레이션 (저주파 + 중간주파)
    engine_base = np.sin(2 * np.pi * 80 * t)  # 80Hz 기본 엔진 주파수
    engine_harmonics = 0.3 * np.sin(2 * np.pi * 160 * t)  # 2차 하모닉
    noise = 0.1 * np.random.normal(0, 1, len(t))  # 약간의 노이즈
    
    fake_audio = engine_base + engine_harmonics + noise
    fake_audio = fake_audio.astype(np.float32)
    
    print("가짜 엔진 소리 데이터로 분석 중...")
    result = system._analyze_audio_data(fake_audio, sample_rate)
    system.print_diagnosis_report(result)

def test_high_frequency_noise():
    """고주파 노이즈 테스트 (베어링 문제 시뮬레이션)"""
    print("\n🧪 고주파 노이즈 테스트")
    
    system = CarNoiseDiagnosisSystem()
    
    duration = 2.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 베어링 문제 시뮬레이션 (고주파 성분 많음)
    bearing_noise = 0.5 * np.sin(2 * np.pi * 2000 * t)  # 2kHz
    high_freq_noise = 0.3 * np.sin(2 * np.pi * 4000 * t)  # 4kHz
    random_noise = 0.2 * np.random.normal(0, 1, len(t))
    
    fake_audio = bearing_noise + high_freq_noise + random_noise
    fake_audio = fake_audio.astype(np.float32)
    
    print("베어링 문제 시뮬레이션 데이터로 분석 중...")
    result = system._analyze_audio_data(fake_audio, sample_rate)
    system.print_diagnosis_report(result)

if __name__ == "__main__":
    print("🚗 자동차 소음 진단 시스템 테스트")
    print("="*50)
    
    try:
        test_basic_functionality()
        test_high_frequency_noise()
        
        print("\n✅ 모든 테스트 완료!")
        print("실제 사용을 위해서는 python main.py를 실행하세요.")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        print("requirements.txt의 패키지들이 설치되어 있는지 확인해보세요.")