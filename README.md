# 🚗 자동차 소음 진단 시스템

MediaPipe Audio Classifier를 기반으로 한 실시간 자동차 소음 분석 및 고장 진단 시스템입니다.

## ✨ 주요 기능

- **실시간 오디오 분석**: 마이크를 통한 실시간 자동차 소음 캡처 및 분석
- **오디오 파일 분석**: 녹음된 오디오 파일 분석 지원
- **AI 기반 분류**: MediaPipe의 YAMNet 모델을 사용한 소음 분류
- **자동차 특화 진단**: 엔진, 브레이크, 베어링 등 부품별 상태 진단
- **한국어 인터페이스**: 직관적인 한국어 진단 결과 제공

## 🏗️ 프로젝트 구조

```
noise-diagnosis/
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   └── capture.py          # 오디오 입력 처리
│   ├── models/
│   │   ├── __init__.py
│   │   └── mediapipe_classifier.py  # MediaPipe 분류기
│   ├── diagnosis/
│   │   ├── __init__.py
│   │   └── analyzer.py         # 진단 로직
│   └── __init__.py
├── data/
│   ├── samples/               # 샘플 오디오 파일
│   └── models/               # 커스텀 모델 저장소
├── main.py                   # 메인 실행 파일
├── test_example.py          # 테스트 스크립트
├── requirements.txt         # 의존성 패키지
└── README.md               # 이 파일
```

## 🚀 설치 및 실행

### 1. 가상환경 설정
```bash
# pyenv 사용 시
pyenv virtualenv 3.10.6 noise-diagnosis
pyenv activate noise-diagnosis

# 또는 venv 사용 시
python -m venv noise-diagnosis
source noise-diagnosis/bin/activate  # Linux/Mac
# noise-diagnosis\\Scripts\\activate  # Windows
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 실행
```bash
# 메인 프로그램 실행
python main.py

# 테스트 스크립트 실행
python test_example.py
```

## 📋 시스템 요구사항

- Python 3.8 이상
- macOS/Linux (라즈베리파이 호환)
- 마이크 (실시간 분석 시)
- 충분한 메모리 (MediaPipe 모델 로딩)

## 🔧 주요 모듈 설명

### AudioCapture (`src/audio/capture.py`)
- 실시간 오디오 캡처
- 오디오 파일 로딩
- 다중 스레드 안전한 오디오 버퍼링

### MediaPipeAudioClassifier (`src/models/mediapipe_classifier.py`)
- MediaPipe YAMNet 모델 래핑
- 자동차 관련 소음 필터링
- 오디오 특성 추출 (MFCC, 스펙트럴 특성 등)

### CarNoiseDiagnoser (`src/diagnosis/analyzer.py`)
- 규칙 기반 진단 엔진
- 부품별 상태 분류
- 신뢰도 계산 및 추천사항 생성

## 🎯 진단 가능한 부품

- **엔진**: 저주파 노이즈, 비정상 진동
- **브레이크**: 고주파 소음, 마찰음
- **베어링**: 고주파 소음, 마모음
- **벨트**: 슬립음, 스펙트럴 이상
- **타이어**: 마찰음, 로드 노이즈
- **배기계통**: 배기음 이상
- **변속기**: 기어 변속음

## 📊 출력 예시

```
🚗 자동차 소음 진단 결과
============================================================
📊 분석 정보:
   - 분석 시간: 5.0초
   - 샘플링 레이트: 16000 Hz
   - 음량 레벨: 0.125
   - 진단 신뢰도: 78.5%

🎯 전체 상태: ⚠️  주의

🔊 감지된 차량 관련 소리:
   - 엔진: Engine idling (85.2%)
   - 베어링: Mechanical noise (42.1%)

🔍 발견된 문제들:
   1. ⚠️  [베어링] 고주파 노이즈 증가 - 베어링 마모 가능성
      신뢰도: 73.4%

💡 추천 사항:
   1. ⚠️  가까운 시일 내 점검을 받으시기 바랍니다:
   2. - 고주파 노이즈 증가 - 베어링 마모 가능성
```

## 🔮 향후 개발 계획

1. **커스텀 모델 훈련**
   - 자동차별 특화 모델
   - 더 정확한 부품별 진단

2. **라즈베리파이 최적화**
   - 경량화된 모델
   - 실시간 성능 개선

3. **데이터 수집 및 학습**
   - 실제 차량 소음 데이터셋 구축
   - 지도학습 기반 정확도 향상

4. **UI 개선**
   - 웹 인터페이스 추가
   - 진단 이력 관리

## 🤝 기여 방법

1. 실제 차량 소음 데이터 제공
2. 새로운 진단 규칙 제안
3. 코드 개선 및 버그 리포트
4. 문서화 개선

## 📝 라이센스

이 프로젝트는 교육 목적으로 개발되었습니다.

## ⚠️  주의사항

- 이 시스템은 참고용이며, 실제 정비 결정은 전문가와 상담하세요
- 실시간 분석 시 마이크 권한이 필요합니다
- MediaPipe 모델 로딩에 시간이 걸릴 수 있습니다