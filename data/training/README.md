# 🎵 학습 데이터 수집 가이드

이 폴더는 자동차 소음 분류 모델을 학습시키기 위한 데이터를 저장하는 곳입니다.

## 📂 폴더 구조

각 하위 폴더는 하나의 클래스(라벨)를 나타냅니다:

```
training/
├── normal/              # 정상적인 자동차 소음
├── engine_problem/      # 엔진 문제 (노킹, 미스파이어 등)
├── brake_problem/       # 브레이크 문제 (삐걱거림, 마찰음 등)
├── bearing_problem/     # 베어링 문제 (마모음, 그라인딩 등)
├── belt_problem/        # 벨트 문제 (슬립, 삐걱거림 등)
├── tire_problem/        # 타이어 문제 (불균형, 편마모 등)
└── transmission_problem/ # 변속기 문제 (기어 변속 이상 등)
```

## 🎯 데이터 수집 방법

### 1. YouTube에서 수집

**추천 검색어:**
- Normal: "car idle sound", "engine sound normal"
- Engine: "engine knocking sound", "car misfire sound"
- Brake: "brake squeal sound", "brake noise"
- Bearing: "wheel bearing noise", "bad bearing sound car"
- Belt: "serpentine belt noise", "belt squeal"
- Tire: "tire noise", "wheel imbalance sound"
- Transmission: "transmission problem sound", "gear shifting noise"

**다운로드 방법:**
```bash
# yt-dlp 설치 (Mac)
brew install yt-dlp

# 오디오만 다운로드 (WAV 형식)
yt-dlp -x --audio-format wav -o "normal/%(title)s.%(ext)s" "https://youtube.com/watch?v=..."

# 재생목록 전체 다운로드
yt-dlp -x --audio-format wav -o "engine_problem/%(title)s.%(ext)s" "https://youtube.com/playlist?list=..."
```

### 2. Kaggle 데이터셋

**검색 키워드:**
- "car sound dataset"
- "vehicle noise dataset"
- "automotive audio classification"

다운로드 후 적절한 폴더로 분류하세요.

### 3. FreeSound.org

무료 효과음 라이브러리:
- https://freesound.org/search/?q=car+engine
- https://freesound.org/search/?q=brake+squeal
- 회원가입 후 다운로드 가능

### 4. 직접 녹음

**필요한 장비:**
- 스마트폰 또는 녹음기
- 가능하면 외부 마이크 (더 깨끗한 음질)

**녹음 팁:**
- 샘플링 레이트: 16kHz 이상 (48kHz 권장)
- 길이: 3-10초 (너무 짧지 않게)
- 배경 소음 최소화
- 다양한 RPM, 속도에서 녹음

## 📊 권장 데이터량

| 목적 | 각 클래스당 샘플 수 |
|------|-------------------|
| 최소 동작 | 10개 이상 |
| 기본 학습 | 30-50개 |
| 권장 | 50-100개 |
| 이상적 | 100개 이상 |

**현재 상태 확인:**
```bash
# 각 폴더별 파일 개수 확인
for dir in */; do echo "$dir: $(find "$dir" -type f | wc -l) files"; done
```

## 🎵 지원 오디오 형식

- WAV (추천)
- MP3
- MP4/M4A
- FLAC
- OGG

## 🔧 데이터 전처리

`train.py`가 자동으로 수행:
- 모노 변환 (스테레오 → 모노)
- 리샘플링 (16kHz로 통일)
- YAMNet 특성 추출 (88차원 벡터)

## 📝 파일 이름 규칙

명확한 파일명 권장:
```
✅ 좋은 예:
- normal/toyota_camry_idle_normal.wav
- engine_problem/honda_civic_knocking.mp3
- brake_problem/ford_focus_brake_squeal.wav

❌ 피할 예:
- audio1.wav
- test.mp3
- recording (1).wav
```

## 🚀 학습 시작

데이터 준비가 완료되면:

```bash
# 프로젝트 루트로 이동
cd ../..

# 모델 학습
python train.py

# 결과 확인
python evaluate.py
```

## 💡 데이터 품질 향상 팁

1. **다양성 확보**
   - 다양한 차종
   - 다양한 녹음 환경
   - 다양한 문제 심각도

2. **균형 유지**
   - 각 클래스 데이터량 비슷하게
   - 편향 방지

3. **노이즈 최소화**
   - 배경 음악 없는 소리
   - 음성 나레이션 최소화
   - 깨끗한 오디오 선호

4. **검증**
   - 파일이 제대로 재생되는지 확인
   - 손상된 파일 제거
   - 올바른 폴더에 분류되었는지 확인

## 📚 유용한 리소스

- **YouTube Channels:**
  - "Car Wizard" (차량 문제 진단)
  - "Scanner Danner" (자동차 정비)
  - "South Main Auto" (실제 정비 영상)

- **Datasets:**
  - Urban Sound 8K (일부 차량음 포함)
  - AudioSet (Google, 차량 관련 태그)

- **Forums:**
  - r/MechanicAdvice (Reddit)
  - Car Talk Community

## ⚠️ 주의사항

- 저작권: YouTube 동영상 사용 시 저작권 확인
- 교육 목적: 본 프로젝트는 교육/연구 목적
- 개인정보: 녹음 시 개인정보 포함 안 되도록 주의

## 🆘 문제 해결

**문제: 데이터가 너무 적음**
→ 데이터 증강 사용 (피치 변환, 속도 조절 등)
→ Transfer Learning 고려

**문제: 클래스 불균형**
→ 적은 클래스 데이터 추가 수집
→ 또는 가중치 조정

**문제: 정확도가 낮음**
→ 데이터 품질 확인
→ 잘못 분류된 파일 재확인
→ 더 많은 데이터 수집
