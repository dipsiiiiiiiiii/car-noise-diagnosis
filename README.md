# 🚗 자동차 소음 진단 시스템

YAMNet 기반 딥러닝 모델을 활용한 지능형 자동차 소음 분석 및 고장 진단 시스템입니다.

## ✨ 주요 기능

- **실시간 오디오 분석**: 마이크를 통한 실시간 자동차 소음 캡처 및 분석
- **오디오 파일 분석**: 녹음된 오디오 파일 분석 지원
- **이중 진단 시스템**:
  - **Baseline 모드**: YAMNet 범용 모델만 사용 (521개 클래스)
  - **Custom 모드**: 자동차 특화 학습 모델 사용 (7개 클래스)
  - **비교 모드**: 두 모델을 동시에 실행하여 실시간 비교 (발표/시연용)
- **데이터 기반 학습**: 실제 데이터로 학습 가능한 확장형 구조
- **성능 비교 분석**: Baseline vs Custom 모델 성능 비교 그래프 생성
- **한국어 인터페이스**: 직관적인 한국어 진단 결과 제공

## 🏗️ 프로젝트 구조

```
noise-diagnosis/
├── src/
│   ├── audio/
│   │   ├── __init__.py
│   │   └── capture.py                  # 오디오 입력 처리
│   ├── models/
│   │   ├── __init__.py
│   │   └── mediapipe_classifier.py    # YAMNet 분류기 + 임베딩 추출
│   ├── diagnosis/
│   │   ├── __init__.py
│   │   └── analyzer.py                # 진단 로직 (Baseline + Custom)
│   └── __init__.py
├── data/
│   ├── training/                      # 학습 데이터
│   │   ├── normal/                   # 정상 소리
│   │   ├── engine_problem/          # 엔진 문제
│   │   ├── brake_problem/           # 브레이크 문제
│   │   ├── bearing_problem/         # 베어링 문제
│   │   ├── belt_problem/            # 벨트 문제
│   │   ├── tire_problem/            # 타이어 문제
│   │   └── transmission_problem/    # 변속기 문제
│   ├── models/
│   │   ├── yamnet.tflite            # YAMNet 모델
│   │   └── car_classifier.pkl       # 학습된 커스텀 모델
│   ├── evaluation_results/          # 평가 결과 그래프
│   └── samples/                     # 샘플 오디오
├── main.py                           # 메인 진단 프로그램
├── train.py                          # 모델 학습 스크립트
├── evaluate.py                       # 모델 평가 및 비교 스크립트
├── test_example.py                   # 테스트 스크립트
├── requirements.txt                  # 의존성 패키지
└── README.md                         # 이 파일
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

### 3. YAMNet 모델 다운로드
```bash
curl -L 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite' -o data/models/yamnet.tflite
```

### 4. 실행

```bash
# 프로그램 실행
python main.py

# 시작 시 옵션:
# - 디버그 모드: YAMNet 분류 상세 결과 표시
# - 비교 모드: Baseline과 Custom을 동시에 비교 (Custom 모델 있을 때만)
```

**동작 모드:**
- **Custom 모델 없음** → Baseline 모드 (YAMNet만)
- **Custom 모델 있음** → Custom 모드 (학습된 모델 사용)
- **비교 모드 활성화** → 두 모델 동시 실행 및 비교 출력

## 📋 시스템 요구사항

- Python 3.8 이상
- macOS/Linux (라즈베리파이 호환)
- 마이크 (실시간 분석 시)
- 충분한 메모리 (MediaPipe 모델 로딩)

## 🔧 주요 모듈 설명

### AudioCapture (`src/audio/capture.py`)
- 실시간 오디오 캡처 (ffmpeg 기반)
- 오디오 파일 로딩 및 전처리
- 다중 스레드 안전한 오디오 버퍼링

### MediaPipeAudioClassifier (`src/models/mediapipe_classifier.py`)
- YAMNet 모델 래핑 (521개 범용 클래스)
- 자동차 관련 소음 필터링
- **특성 벡터 추출** (88차원):
  - YAMNet Top-50 예측 확률 (50차원)
  - 오디오 특성 (12차원): RMS, spectral centroid, rolloff 등
  - MFCC 특성 (26차원): 13 mean + 13 std

### CarNoiseDiagnoser (`src/diagnosis/analyzer.py`)
- **Baseline 모드**: YAMNet 키워드 기반 휴리스틱
- **Custom 모드**: 학습된 Random Forest 분류기 사용
- 7개 클래스 진단:
  1. 정상
  2. 엔진 문제
  3. 브레이크 문제
  4. 베어링 문제
  5. 벨트 문제
  6. 타이어 문제
  7. 변속기 문제

## 📊 출력 예시

### 일반 모드 (Custom 또는 Baseline)
```
🚗 자동차 소음 진단 결과
============================================================
📊 분석 정보:
   - 분석 시간: 5.0초
   - 샘플링 레이트: 16000 Hz
   - 음량 레벨: 0.125
   - 진단 신뢰도: 87.5%

🎯 전체 상태: ⚠️  주의

🔍 발견된 문제들:
   1. ⚠️ [엔진] 엔진 이상 감지 (Custom Model: 87.5%)
      신뢰도: 87.5%

💡 추천 사항:
   1. ⚠️ 가까운 시일 내 점검을 받으시기 바랍니다
```

### 비교 모드 (발표/시연용)
```
🚗 자동차 소음 진단 결과 (비교 모드)
======================================================================
📊 분석 정보:
   - 분석 시간: 5.0초
   - 샘플링 레이트: 16000 Hz
   - 음량 레벨: 0.125

----------------------------------------------------------------------
📌 YAMNet Baseline (범용 모델)
----------------------------------------------------------------------
   상태: ⚠️ 주의
   신뢰도: 45.2%
   문제:
     - [베어링] 마모음 감지 (YAMNet: 45.2%)

----------------------------------------------------------------------
🎯 Custom Model (자동차 특화 학습 모델)
----------------------------------------------------------------------
   상태: ⚠️ 주의
   신뢰도: 87.5%
   문제:
     - [엔진] 엔진 이상 감지 (Custom Model: 87.5%)

----------------------------------------------------------------------
📈 비교 분석
----------------------------------------------------------------------
   ✅ Custom 모델이 42.3%p 더 확신합니다
   ⚠️ 두 모델의 진단이 다릅니다

💡 추천:
   → Custom 모델 결과를 우선적으로 참고하세요 (신뢰도 높음)
======================================================================
```

## 📚 데이터 수집 및 학습

### 1. 데이터 수집

자동차 소음 오디오 파일을 수집하여 `data/training/` 폴더에 분류:

```bash
# 예시 데이터 구조
data/training/
  ├── normal/
  │   ├── car_idle_normal_1.wav
  │   ├── car_idle_normal_2.mp3
  │   └── ...
  ├── engine_problem/
  │   ├── engine_knocking_1.wav
  │   ├── engine_misfire_2.mp3
  │   └── ...
  ├── brake_problem/
  │   ├── brake_squeal_1.wav
  │   └── ...
  └── ...
```

**데이터 수집 방법:**
1. **Kaggle**: 자동차 소음 데이터셋 검색
2. **YouTube**: 차량 문제 진단 영상에서 오디오 추출
   ```bash
   # yt-dlp로 YouTube 오디오 다운로드
   yt-dlp -x --audio-format wav "https://youtube.com/watch?v=..."
   ```
3. **FreeSound**: 효과음 라이브러리 활용
4. **직접 녹음**: 실제 차량에서 녹음

**권장 데이터량:**
- 최소: 각 클래스당 10개 이상
- 권장: 각 클래스당 50개 이상
- 이상적: 각 클래스당 100개 이상

### 2. 모델 학습

```bash
# 데이터가 준비되면 학습 실행
python train.py

# 출력 예시:
# ✅ 총 350개 샘플 로드 완료
# 훈련 데이터: 280개
# 테스트 데이터: 70개
# 테스트 정확도: 92.5%
# ✅ 모델 저장 완료: data/models/car_classifier.pkl
```

### 3. 모델 평가 및 비교

```bash
# Baseline vs Custom 모델 성능 비교
python evaluate.py

# 생성되는 그래프:
# - accuracy_comparison.png        (정확도 비교)
# - confusion_matrices.png         (혼동 행렬)
# - per_class_performance.png      (클래스별 성능)
# - confidence_distribution.png    (신뢰도 분포)
```

생성된 그래프는 `data/evaluation_results/`에 저장되며, 발표 자료로 활용 가능합니다.

### 4. 진단 사용

```bash
# 학습된 모델로 진단
python main.py

# 출력:
# ✅ 학습된 모델 로드됨: data/models/car_classifier.pkl
# 🚗 자동차 소음 진단 중... (모드: custom)
```

## 🔮 향후 개발 계획

1. **모델 개선**
   - YAMNet Fine-tuning으로 더 높은 정확도
   - LSTM/CNN으로 시계열 패턴 학습

2. **라즈베리파이 최적화**
   - TFLite 모델 경량화
   - 실시간 추론 속도 개선

3. **데이터 증강**
   - 오디오 노이즈 추가
   - 피치 변환, 시간 늘이기 등

4. **UI 개선**
   - 웹 인터페이스 추가
   - 진단 이력 관리
   - 실시간 스펙트로그램 시각화

## 📝 라이센스

이 프로젝트는 교육 목적으로 개발되었습니다.

## ⚠️  주의사항

- 이 시스템은 참고용이며, 실제 정비 결정은 전문가와 상담하세요
- 실시간 분석 시 마이크 권한이 필요합니다
- MediaPipe 모델 로딩에 시간이 걸릴 수 있습니다