#!/usr/bin/env python3
"""
AudioSet에서 엔진 소리 데이터 다운로드
- 정상 엔진 소리 (Idling, Medium engine)
- 노킹 소리 (Engine knocking) - 추가 데이터
"""

import os
from pathlib import Path

print("="*80)
print("🎵 AudioSet 엔진 소리 데이터 다운로드")
print("="*80)

# 출력 디렉토리 설정
output_base = Path("data/training/audioset")
output_base.mkdir(parents=True, exist_ok=True)

# 다운로드할 카테고리 정의
categories = {
    'normal_idling': '/m/07pb8fc',      # Idling (정상 공회전)
    'normal_medium': '/m/08j51y',       # Medium engine (정상 주행)
    'knocking': '/m/07pdhp0'            # Engine knocking (노킹)
}

print("\n📋 다운로드 계획:")
print(f"   1. Idling (공회전) → {output_base}/normal_idling/")
print(f"   2. Medium engine (주행) → {output_base}/normal_medium/")
print(f"   3. Engine knocking (노킹) → {output_base}/knocking/")
print()

# audioset-download 설치 확인
try:
    import audioset_download
    print("✅ audioset-download 패키지 확인됨")
except ImportError:
    print("❌ audioset-download가 설치되어 있지 않습니다.")
    print("\n설치 방법:")
    print("   pip install audioset-download")
    exit(1)

print("\n" + "="*80)
print("⚠️  AudioSet 다운로드 방법")
print("="*80)
print("""
AudioSet은 YouTube 영상에서 추출한 데이터셋입니다.
다음 두 가지 방법으로 다운로드할 수 있습니다:

방법 1: audioset-download CLI 사용
--------------------------------------
# 정상 엔진 소리 (Idling)
audioset-download \\
    --label-name "Idling" \\
    --output-dir data/training/audioset/normal_idling \\
    --num-workers 4 \\
    --max-duration 5 \\
    --num-samples 200

# 정상 엔진 소리 (Medium engine)
audioset-download \\
    --label-name "Medium engine (mid frequency)" \\
    --output-dir data/training/audioset/normal_medium \\
    --num-workers 4 \\
    --max-duration 5 \\
    --num-samples 200

# 노킹 소리 (추가 데이터)
audioset-download \\
    --label-name "Engine knocking" \\
    --output-dir data/training/audioset/knocking \\
    --num-workers 4 \\
    --max-duration 5 \\
    --num-samples 100


방법 2: 직접 YouTube 영상에서 추출
--------------------------------------
노킹이 없는 정상 엔진 소리 영상을 YouTube에서 찾아서
extract_manual_range.py로 추출할 수도 있습니다.

검색 키워드:
- "healthy engine sound idle"
- "normal car engine running"
- "smooth engine idle"


다운로드 후 처리:
--------------------------------------
1. 다운로드한 오디오를 3초 세그먼트로 분할
2. 너무 조용한 샘플 제거
3. data/training/normal/ 에 정리

그 다음:
4. train_two_class.py (새로 만들 예정)로 학습
""")

print("\n" + "="*80)
print("💡 다음 단계")
print("="*80)
print("""
1. 위 명령어로 AudioSet 데이터 다운로드
2. 또는 YouTube에서 직접 수집
3. 정상/노킹 데이터 정리 완료 후:
   python train_two_class.py  # 2클래스 분류기 학습
""")
print("="*80)
