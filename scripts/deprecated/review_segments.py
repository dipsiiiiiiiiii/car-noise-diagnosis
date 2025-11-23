#!/usr/bin/env python3
"""
Manual Review Helper for Knocking Segments
후보 세그먼트를 재생하고 수동으로 검수
"""

import sys
import shutil
from pathlib import Path
import subprocess
import re


def play_audio(audio_path: Path):
    """Play audio file using afplay (macOS)"""
    try:
        subprocess.run(['afplay', str(audio_path)], check=True)
    except FileNotFoundError:
        print("  ⚠️  afplay를 찾을 수 없습니다. macOS에서만 작동합니다.")
        print("  💡 수동으로 파일을 재생하세요.")
    except Exception as e:
        print(f"  ⚠️  재생 오류: {e}")


def extract_info_from_filename(filename: str) -> dict:
    """Extract metadata from filename"""
    # Format: basename_t123.4s_score56_001.wav
    match = re.match(r'(.+)_t(\d+\.\d+)s_score(\d+)_(\d+)\.wav', filename)

    if match:
        return {
            'base': match.group(1),
            'time': float(match.group(2)),
            'score': int(match.group(3)),
            'index': int(match.group(4))
        }
    return {}


def review_segments(candidates_dir: Path, verified_dir: Path, rejected_dir: Path):
    """Interactive review process"""

    # Get all candidates
    candidates = sorted(candidates_dir.glob("*.wav"))

    if not candidates:
        print(f"\n❌ 검수할 후보가 없습니다: {candidates_dir}")
        return

    print(f"\n📊 총 {len(candidates)}개 후보 세그먼트")
    print("\n" + "="*80)
    print("🎧 수동 검수 시작")
    print("="*80)
    print("\n💡 사용법:")
    print("   y = 노킹 맞음 (verified로 이동)")
    print("   n = 노킹 아님 (rejected로 이동)")
    print("   r = 다시 재생")
    print("   s = 건너뛰기")
    print("   q = 종료")
    print("="*80)

    verified_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, candidate in enumerate(candidates, 1):
        # Extract info
        info = extract_info_from_filename(candidate.name)

        print(f"\n[{i}/{len(candidates)}] {candidate.name}")

        if info:
            print(f"   원본: {info['base']}")
            print(f"   시간: {info['time']:.1f}초")
            print(f"   YAMNet 노킹 점수: {info['score']}%")

        # Play audio
        print("   🔊 재생 중...")
        play_audio(candidate)

        # Get user decision
        while True:
            choice = input("\n   노킹인가요? [y/n/r/s/q]: ").strip().lower()

            if choice == 'y':
                # Move to verified
                dest = verified_dir / candidate.name
                shutil.move(str(candidate), str(dest))
                print(f"   ✅ Verified로 이동")
                verified_count += 1
                break

            elif choice == 'n':
                # Move to rejected
                dest = rejected_dir / candidate.name
                shutil.move(str(candidate), str(dest))
                print(f"   ❌ Rejected로 이동")
                rejected_count += 1
                break

            elif choice == 'r':
                # Replay
                print("   🔊 다시 재생 중...")
                play_audio(candidate)
                continue

            elif choice == 's':
                # Skip
                print(f"   ⏭️  건너뜀")
                skipped_count += 1
                break

            elif choice == 'q':
                # Quit
                print("\n👋 검수 중단")
                print_summary(verified_count, rejected_count, skipped_count, i-1, len(candidates))
                return

            else:
                print("   ⚠️  잘못된 입력입니다. y/n/r/s/q 중 선택하세요.")

    # Final summary
    print("\n" + "="*80)
    print("✅ 검수 완료!")
    print_summary(verified_count, rejected_count, skipped_count, len(candidates), len(candidates))


def print_summary(verified, rejected, skipped, reviewed, total):
    """Print review summary"""
    print("="*80)
    print(f"📊 검수 결과:")
    print(f"   - 검수 완료: {reviewed}/{total}개")
    print(f"   - ✅ Verified (노킹): {verified}개")
    print(f"   - ❌ Rejected (노킹 아님): {rejected}개")
    print(f"   - ⏭️  Skipped (건너뜀): {skipped}개")

    if verified > 0:
        print(f"\n💡 다음 단계:")
        print(f"   python train_verified.py  # 검수된 데이터로 학습")

    print("="*80)


def main():
    print("="*80)
    print("🔍 노킹 세그먼트 수동 검수")
    print("="*80)

    # Paths
    candidates_dir = Path("data/training/manual_workflow/1_candidates")
    verified_dir = Path("data/training/manual_workflow/2_verified")
    rejected_dir = Path("data/training/manual_workflow/3_rejected")

    # Create directories
    verified_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    # Check candidates
    if not candidates_dir.exists():
        print(f"\n❌ 후보 폴더를 찾을 수 없습니다: {candidates_dir}")
        print("먼저 python extract_candidates.py를 실행하세요.")
        return

    # Start review
    review_segments(candidates_dir, verified_dir, rejected_dir)


if __name__ == "__main__":
    main()
