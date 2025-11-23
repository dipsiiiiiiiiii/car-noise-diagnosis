#!/usr/bin/env python3
"""
AudioSet에서 엔진 소리 데이터 다운로드
- 정상: Idling, Medium engine
- 노킹: Engine knocking (추가 데이터)
"""

from audioset_download import Downloader
from pathlib import Path
import shutil

print("="*80)
print("🎵 AudioSet 엔진 소리 다운로드")
print("="*80)

# 출력 디렉토리
output_root = Path("data/training/audioset")
output_root.mkdir(parents=True, exist_ok=True)

print("\n📋 다운로드할 카테고리:")
print("   1. Idling (공회전) - 정상")
print("   2. Medium engine (mid frequency) - 정상")
print("   3. Engine knocking - 노킹 (추가)")

# Step 1: Download Idling (정상 공회전)
print("\n" + "="*80)
print("📥 [1/3] Idling (공회전) 다운로드")
print("="*80)

try:
    downloader_idling = Downloader(
        root_path=str(output_root / "temp_idling"),
        labels=["Idling"],
        n_jobs=4,
        download_type='unbalanced_train'
    )
    downloader_idling.download(format='wav', quality=0)
    print("✅ Idling 다운로드 완료")
except Exception as e:
    print(f"⚠️  Idling 다운로드 오류: {e}")

# Step 2: Download Medium engine (정상 주행)
print("\n" + "="*80)
print("📥 [2/3] Medium engine 다운로드")
print("="*80)

try:
    downloader_medium = Downloader(
        root_path=str(output_root / "temp_medium"),
        labels=["Medium engine (mid frequency)"],
        n_jobs=4,
        download_type='unbalanced_train'
    )
    downloader_medium.download(format='wav', quality=0)
    print("✅ Medium engine 다운로드 완료")
except Exception as e:
    print(f"⚠️  Medium engine 다운로드 오류: {e}")

# Step 3: Download Engine knocking (노킹)
print("\n" + "="*80)
print("📥 [3/3] Engine knocking 다운로드")
print("="*80)

try:
    downloader_knocking = Downloader(
        root_path=str(output_root / "temp_knocking"),
        labels=["Engine knocking"],
        n_jobs=4,
        download_type='unbalanced_train'
    )
    downloader_knocking.download(format='wav', quality=0)
    print("✅ Engine knocking 다운로드 완료")
except Exception as e:
    print(f"⚠️  Engine knocking 다운로드 오류: {e}")

# Reorganize files
print("\n" + "="*80)
print("📁 파일 정리 중...")
print("="*80)

normal_dir = Path("data/training/normal")
knocking_dir = Path("data/training/audioset_knocking")

normal_dir.mkdir(parents=True, exist_ok=True)
knocking_dir.mkdir(parents=True, exist_ok=True)

# Move Idling to normal
idling_path = output_root / "temp_idling" / "Idling"
if idling_path.exists():
    for wav_file in idling_path.glob("*.wav"):
        shutil.copy(wav_file, normal_dir / f"idling_{wav_file.name}")
    n_idling = len(list(idling_path.glob("*.wav")))
    print(f"✅ Idling: {n_idling}개 → {normal_dir}")
else:
    n_idling = 0
    print(f"⚠️  Idling 파일 없음")

# Move Medium engine to normal
medium_path = output_root / "temp_medium" / "Medium engine (mid frequency)"
if medium_path.exists():
    for wav_file in medium_path.glob("*.wav"):
        shutil.copy(wav_file, normal_dir / f"medium_{wav_file.name}")
    n_medium = len(list(medium_path.glob("*.wav")))
    print(f"✅ Medium engine: {n_medium}개 → {normal_dir}")
else:
    n_medium = 0
    print(f"⚠️  Medium engine 파일 없음")

# Move Engine knocking
knocking_path = output_root / "temp_knocking" / "Engine knocking"
if knocking_path.exists():
    for wav_file in knocking_path.glob("*.wav"):
        shutil.copy(wav_file, knocking_dir / wav_file.name)
    n_knocking = len(list(knocking_path.glob("*.wav")))
    print(f"✅ Engine knocking: {n_knocking}개 → {knocking_dir}")
else:
    n_knocking = 0
    print(f"⚠️  Engine knocking 파일 없음")

# Cleanup temp directories
print("\n🧹 임시 파일 정리...")
for temp_dir in output_root.glob("temp_*"):
    shutil.rmtree(temp_dir, ignore_errors=True)

# Summary
print("\n" + "="*80)
print("✅ 다운로드 완료!")
print("="*80)
print(f"📊 다운로드된 파일:")
print(f"   정상 (Idling): {n_idling}개")
print(f"   정상 (Medium): {n_medium}개")
print(f"   정상 합계: {n_idling + n_medium}개 → {normal_dir}")
print(f"   노킹: {n_knocking}개 → {knocking_dir}")
print(f"\n💡 다음 단계:")
print(f"   python train_two_class.py  # Two-Class 모델 학습")
print("="*80)
