#!/usr/bin/env python3
"""
루트 디렉토리 스크립트 정리
사용중인 스크립트 / 테스트 / 폐기 로 분류
"""

from pathlib import Path
import shutil

print("="*80)
print("🧹 루트 디렉토리 스크립트 정리")
print("="*80)

# 스크립트 분류
scripts = {
    'active': {
        'description': '현재 사용중인 핵심 스크립트',
        'files': [
            'main.py',                      # 메인 프로그램
            'train_two_class.py',           # 현재 학습 스크립트
            'download_normal_youtube.py',   # 정상 소리 다운로드
            'extract_normal_segments.py',   # 정상 구간 추출
            'augment_normal_sounds.py',     # 정상 데이터 증강
            'augment_knocking_sounds.py',   # 노킹 데이터 증강
            'reorganize_data.py',           # 데이터 폴더 정리
        ]
    },
    'utility': {
        'description': '유틸리티 스크립트 (가끔 사용)',
        'files': [
            'download_audioset_limited.py', # AudioSet 다운로드
            'visualize_augmentation.py',    # 증강 시각화
            'visualize_augmentation_spectrogram.py',  # 스펙트로그램
        ],
        'move_to': 'scripts/utils'
    },
    'testing': {
        'description': '테스트 및 개발 스크립트',
        'files': [
            'test_example.py',
            'test_main_file_analysis.py',
            'test_model_switch.py',
            'test_normal_vs_knocking.py',
            'test_oneclass.py',
            'test_realtime.py',
            'test_verified_model.py',
            'test_video_segments.py',
            'batch_test_videos.py',
            'compare_filtering.py',
            'realtime_filter_comparison.py',
            'evaluate.py',
        ],
        'move_to': 'scripts/tests'
    },
    'deprecated': {
        'description': '더 이상 사용하지 않는 구버전',
        'files': [
            'train.py',                     # 구버전 학습 스크립트
            'train_one_class.py',           # One-Class SVM (안씀)
            'train_one_class_v4.py',        # One-Class v4 (안씀)
            'train_verified.py',            # 구버전
            'augment_data.py',              # 구버전 증강 스크립트
            'extract_candidates.py',        # 구버전 추출
            'extract_knocking_segments.py', # 구버전 (v2로 대체)
            'extract_knocking_segments_v2.py',  # 구버전
            'extract_manual_range.py',      # 구버전 (extract_normal_segments로 대체)
            'review_segments.py',           # 구버전 검수 도구
            'quality_check_segments.py',    # 구버전 품질 체크
            'download_audioset.py',         # 구버전
            'download_audioset_data.py',    # 구버전
        ],
        'move_to': 'scripts/deprecated'
    }
}

# 현재 상태 출력
print("\n📂 스크립트 분류:")
print("="*80)

total_files = 0
for category, info in scripts.items():
    existing_files = [f for f in info['files'] if Path(f).exists()]
    total_files += len(existing_files)

    if category == 'active':
        print(f"\n✅ {info['description']} (루트에 유지)")
    else:
        move_to = info.get('move_to', 'unknown')
        print(f"\n📦 {info['description']} → {move_to}/")

    for f in existing_files:
        print(f"   - {f}")

print(f"\n총 {total_files}개 스크립트")

# 확인
print("\n" + "="*80)
print("정리 계획:")
print("  - Active 스크립트: 루트에 유지")
print("  - Utility 스크립트: scripts/utils/로 이동")
print("  - Testing 스크립트: scripts/tests/로 이동")
print("  - Deprecated 스크립트: scripts/deprecated/로 이동")
print("="*80)

import sys
if len(sys.argv) > 1 and sys.argv[1] == '--confirm':
    choice = 'y'
else:
    try:
        choice = input("\n정리를 진행하시겠습니까? (y/N): ").strip().lower()
    except EOFError:
        print("\n사용법: python cleanup_scripts.py --confirm")
        exit(1)

if choice != 'y':
    print("취소되었습니다.")
    exit(0)

# 정리 실행
print("\n⚙️  스크립트 정리 중...")

for category, info in scripts.items():
    if category == 'active':
        continue  # Active는 이동하지 않음

    move_to = Path(info.get('move_to', 'scripts/other'))
    move_to.mkdir(parents=True, exist_ok=True)

    print(f"\n[{category.upper()}] → {move_to}/")

    moved_count = 0
    for filename in info['files']:
        filepath = Path(filename)
        if filepath.exists():
            dest = move_to / filename
            shutil.move(str(filepath), str(dest))
            print(f"  ✅ {filename}")
            moved_count += 1

    print(f"  총 {moved_count}개 이동")

# 결과
print("\n" + "="*80)
print("✅ 스크립트 정리 완료!")
print("="*80)

print("\n📂 루트 디렉토리에 남은 핵심 스크립트:")
for f in scripts['active']['files']:
    if Path(f).exists():
        print(f"   ✅ {f}")

print("\n📦 정리된 스크립트:")
print(f"   - scripts/utils/      : 유틸리티")
print(f"   - scripts/tests/      : 테스트")
print(f"   - scripts/deprecated/ : 폐기 (나중에 삭제 가능)")

print("\n💡 다음 단계:")
print("   1. 루트 디렉토리가 깔끔해졌는지 확인: ls *.py")
print("   2. 워크플로우 진행: WORKFLOW.md 참고")
print("="*80)
