#!/usr/bin/env python3
"""
데이터 폴더 구조 정리
중구난방인 폴더를 깔끔하게 정리
"""

import shutil
from pathlib import Path

print("="*80)
print("📁 데이터 폴더 구조 재정리")
print("="*80)

# 새로운 폴더 구조 정의
new_structure = {
    'raw': {
        'audioset_idling': 'data/training/audioset/Idling',
        'audioset_medium': 'data/training/audioset/Medium_engine_mid_frequency',
        'youtube_normal': 'data/training/youtube_normal',
    },
    'manual_review': {
        'normal_verified': None,  # 아직 없음
        'knocking_candidates': 'data/training/manual_workflow/1_candidates',
        'knocking_verified': 'data/training/manual_workflow/2_verified',
        'knocking_rejected': 'data/training/manual_workflow/3_rejected',
    },
    'processed': {
        'normal_augmented': 'data/training/normal',
        'knocking_augmented': 'data/training/knocking_augmented',
    },
    'deprecated': {
        # 더 이상 사용하지 않는 폴더들
        'engine_knocking': 'data/training/engine_knocking',
        'engine_knocking_augmented': 'data/training/engine_knocking_augmented',
        'engine_knocking_segments': 'data/training/engine_knocking_segments',
        'engine_knocking_segments_v2': 'data/training/engine_knocking_segments_v2',
    }
}

# 새 폴더 구조 생성
base_dir = Path("data/training")
new_base = {
    'raw': base_dir / "raw",
    'manual_review': base_dir / "manual_review",
    'processed': base_dir / "processed",
    'deprecated': base_dir / "_deprecated"  # 백업용
}

print("\n📂 새로운 폴더 구조:")
print("""
data/training/
├── raw/                          # 원본 다운로드 데이터
│   ├── audioset/
│   │   ├── idling/              # AudioSet Idling
│   │   └── medium/              # AudioSet Medium
│   └── youtube/
│       ├── normal/              # YouTube 정상 소리
│       └── knocking/            # YouTube 노킹 소리
├── manual_review/                # 수동 검수 작업 공간
│   ├── normal/
│   │   ├── 1_candidates/        # 자동 추출 후보
│   │   ├── 2_verified/          # 검수 완료
│   │   └── 3_rejected/          # 기각
│   └── knocking/
│       ├── 1_candidates/
│       ├── 2_verified/
│       └── 3_rejected/
├── processed/                    # 최종 학습용 증강 데이터
│   ├── normal/
│   └── knocking/
└── _deprecated/                  # 백업 (나중에 삭제 가능)
""")

import sys
if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
    choice = 'y'
else:
    try:
        choice = input("\n정리를 진행하시겠습니까? (y/N): ").strip().lower()
    except EOFError:
        print("\n사용법: python reorganize_data.py --confirm")
        exit(1)

if choice != 'y':
    print("취소되었습니다.")
    exit(0)

print("\n⚙️  폴더 재구성 중...")

# 1. 새 폴더 생성
for category, path in new_base.items():
    path.mkdir(parents=True, exist_ok=True)

# Raw 데이터 이동
print("\n[1/4] Raw 데이터 이동...")
raw_dir = new_base['raw']

# AudioSet
(raw_dir / "audioset").mkdir(exist_ok=True)
if Path("data/training/audioset/Idling").exists():
    shutil.move("data/training/audioset/Idling", str(raw_dir / "audioset/idling"))
    print("  ✅ AudioSet Idling → raw/audioset/idling/")

if Path("data/training/audioset/Medium_engine_mid_frequency").exists():
    shutil.move("data/training/audioset/Medium_engine_mid_frequency", str(raw_dir / "audioset/medium"))
    print("  ✅ AudioSet Medium → raw/audioset/medium/")

# YouTube
(raw_dir / "youtube").mkdir(exist_ok=True)
if Path("data/training/youtube_normal").exists():
    shutil.move("data/training/youtube_normal", str(raw_dir / "youtube/normal"))
    print("  ✅ YouTube Normal → raw/youtube/normal/")

# Manual Review 이동
print("\n[2/4] Manual Review 데이터 이동...")
review_dir = new_base['manual_review']

# Knocking
(review_dir / "knocking").mkdir(exist_ok=True)
if Path("data/training/manual_workflow/1_candidates").exists():
    shutil.move("data/training/manual_workflow/1_candidates", str(review_dir / "knocking/1_candidates"))
    print("  ✅ Knocking Candidates → manual_review/knocking/1_candidates/")

if Path("data/training/manual_workflow/2_verified").exists():
    shutil.move("data/training/manual_workflow/2_verified", str(review_dir / "knocking/2_verified"))
    print("  ✅ Knocking Verified → manual_review/knocking/2_verified/")

if Path("data/training/manual_workflow/3_rejected").exists():
    shutil.move("data/training/manual_workflow/3_rejected", str(review_dir / "knocking/3_rejected"))
    print("  ✅ Knocking Rejected → manual_review/knocking/3_rejected/")

# Normal (새로 생성)
(review_dir / "normal/1_candidates").mkdir(parents=True, exist_ok=True)
(review_dir / "normal/2_verified").mkdir(parents=True, exist_ok=True)
(review_dir / "normal/3_rejected").mkdir(parents=True, exist_ok=True)
print("  ✅ Normal review 폴더 생성됨")

# Processed 데이터 이동
print("\n[3/4] Processed 데이터 이동...")
processed_dir = new_base['processed']

if Path("data/training/normal").exists():
    shutil.move("data/training/normal", str(processed_dir / "normal"))
    print("  ✅ Normal Augmented → processed/normal/")

if Path("data/training/knocking_augmented").exists():
    shutil.move("data/training/knocking_augmented", str(processed_dir / "knocking"))
    print("  ✅ Knocking Augmented → processed/knocking/")

# Deprecated 폴더 이동
print("\n[4/4] Deprecated 폴더 백업...")
deprecated_dir = new_base['deprecated']

deprecated_folders = [
    "data/training/engine_knocking",
    "data/training/engine_knocking_augmented",
    "data/training/engine_knocking_segments",
    "data/training/engine_knocking_segments_v2",
    "data/training/manual_workflow",
    "data/training/audioset",
]

for folder in deprecated_folders:
    folder_path = Path(folder)
    if folder_path.exists():
        dest = deprecated_dir / folder_path.name
        if not dest.exists():  # 이미 이동한 경우 스킵
            shutil.move(str(folder_path), str(dest))
            print(f"  📦 {folder_path.name} → _deprecated/")

# 결과
print("\n" + "="*80)
print("✅ 폴더 재구성 완료!")
print("="*80)

print("\n📊 새로운 구조:")
import subprocess
try:
    subprocess.run(["tree", "-L", "3", "-d", "data/training"], check=True)
except:
    print("(tree 명령어가 없어 출력을 생략합니다)")

print("\n💡 다음 단계:")
print("   1. python extract_normal_segments.py  # 정상 구간 수동 추출")
print("   2. augment_normal_sounds.py 경로 업데이트")
print("   3. train_two_class.py 경로 업데이트")
print("="*80)
