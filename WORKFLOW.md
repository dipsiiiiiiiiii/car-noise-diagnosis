# 🚗 엔진 노킹 감지 모델 학습 워크플로우

## 📁 데이터 폴더 구조 (재정리 완료)

```
data/training/
├── raw/                          # 원본 데이터
│   ├── audioset/
│   │   ├── idling/              # AudioSet Idling (26개)
│   │   └── medium/              # AudioSet Medium (8개)
│   └── youtube/
│       └── normal/              # YouTube 정상 소리 (6개)
├── manual_review/                # 수동 검수 작업 공간
│   ├── normal/
│   │   ├── 1_candidates/        # 자동 추출 후보
│   │   ├── 2_verified/          # ✅ 검수 완료 (학습용)
│   │   └── 3_rejected/          # ❌ 기각
│   └── knocking/
│       ├── 1_candidates/        # 자동 추출 후보
│       ├── 2_verified/          # ✅ 검수 완료 (172개, 학습용)
│       └── 3_rejected/          # ❌ 기각
├── processed/                    # 증강된 최종 학습 데이터
│   ├── normal/                  # 156개 (Idling 증강)
│   └── knocking/                # 344개 (노킹 증강)
└── _deprecated/                  # 백업 (나중에 삭제 가능)
```

## 🎯 작업 순서

### 1. 정상 소리 구간 수동 추출 ⬅️ **지금 여기!**

YouTube에서 다운로드한 정상 엔진 소리를 듣고 좋은 구간만 추출:

```bash
python extract_normal_segments.py
```

**사용법:**
- `p <번호>` - 파일 전체 재생 (예: p 1)
- `e <번호> <시작> <끝>` - 구간 추출 (예: e 1 5.0 10.0)
- `q` - 종료

**목표:** 정상 소리 구간 50~100개 추출
- 현재: YouTube 6개 파일
- 각 파일에서 5~10개 구간 추출 권장
- 저장 위치: `data/training/manual_review/normal/2_verified/`

### 2. 정상 데이터 증강

추출한 정상 구간을 증강해서 데이터 늘리기:

```bash
python augment_normal_sounds.py
```

**설정:**
- 입력: `data/training/raw/audioset/idling/` + `data/training/manual_review/normal/2_verified/`
- 출력: `data/training/processed/normal/`
- 목표: 300개 이상

### 3. 노킹 데이터 증강 (이미 완료)

현재 상태:
- 원본: 172개 (manual_review/knocking/2_verified/)
- 증강됨: 344개 (processed/knocking/)

### 4. 데이터 확인

```bash
# 정상 데이터 개수
find data/training/manual_review/normal/2_verified -name "*.wav" | wc -l
find data/training/processed/normal -name "*.wav" | wc -l

# 노킹 데이터 개수
find data/training/manual_review/knocking/2_verified -name "*.wav" | wc -l
find data/training/processed/knocking -name "*.wav" | wc -l
```

### 5. 모델 재학습

데이터가 준비되면 모델 학습:

```bash
python train_two_class.py
```

**예상 데이터 분포:**
- 정상: ~300-400개
- 노킹: ~500개 (원본 172 + 증강 344)

### 6. 모델 테스트

```bash
python main.py
```

## 🔧 현재 문제 및 해결책

### 문제: 모델이 항상 70% 노킹으로 예측

**원인:**
- 불균형한 데이터 (정상 38% vs 노킹 62%)
- 실제 원본 정상 소리 부족

**해결책:**
1. ✅ 폴더 구조 정리 완료
2. ✅ YouTube 정상 소리 다운로드 완료 (6개)
3. ⏳ 정상 소리 구간 수동 추출 필요 ← **현재 단계**
4. ⏳ 정상 데이터 증강
5. ⏳ 균형잡힌 데이터로 재학습

## 📝 추가 정상 데이터 수집 방법

1. **YouTube에서 더 다운로드:**
   - `youtube_links_normal.txt`에 링크 추가
   - `python download_normal_youtube.py` 실행

2. **AudioSet에서 다운로드:**
   - `python download_audioset_limited.py` 실행
   - Idling, Medium engine 카테고리

3. **직접 녹음:**
   - 정상적인 자동차 엔진 소리 녹음
   - `data/training/raw/youtube/normal/`에 저장

## 🎬 다음 단계

1. `python extract_normal_segments.py` 실행
2. YouTube 정상 소리 6개 파일을 듣고 좋은 구간 추출 (목표: 50~100개)
3. 데이터 균형 확인
4. 모델 재학습
5. 성능 테스트

## ⚠️ 주의사항

- **youtube_links.txt**: 노킹 링크 (aZjO_FLFnfA는 제외됨 - 정상으로 재분류)
- **youtube_links_normal.txt**: 정상 링크
- **processed/** 폴더의 증강 데이터는 재생성 가능 (원본은 raw/와 manual_review/)
- **_deprecated/** 폴더는 백업이므로 확인 후 삭제 가능
