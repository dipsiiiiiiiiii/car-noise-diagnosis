#!/usr/bin/env python3
"""
AudioSet에서 엔진 소리 제한적 다운로드 (각 카테고리당 200개)
"""

import pandas as pd
import subprocess
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

print("="*80)
print("🎵 AudioSet 엔진 소리 다운로드 (제한: 각 200개)")
print("="*80)

# AudioSet CSV 다운로드
AUDIOSET_CSV_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv"
ONTOLOGY_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

# 라벨 ID
LABELS = {
    'Idling': '/m/07pb8fc',
    'Medium engine (mid frequency)': '/m/08j51y',
    'Engine knocking': '/m/07pdhp0'
}

# 다운로드 제한
MAX_PER_LABEL = 200

# 출력 디렉토리
output_root = Path("data/training/audioset")
output_root.mkdir(parents=True, exist_ok=True)

print("\n📥 AudioSet CSV 다운로드 중...")
csv_path = output_root / "unbalanced_train_segments.csv"

if not csv_path.exists():
    response = requests.get(AUDIOSET_CSV_URL)
    csv_path.write_bytes(response.content)
    print("✅ CSV 다운로드 완료")
else:
    print("✅ CSV 이미 존재함")

# CSV 로드 (skip first 3 comment lines)
print("\n📂 CSV 파싱 중...")
df = pd.read_csv(
    csv_path,
    skiprows=3,
    names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
    skipinitialspace=True
)
print(f"✅ 총 {len(df):,}개 샘플 로드")


def download_segment(ytid, start, end, label_name, output_dir):
    """단일 세그먼트 다운로드"""
    output_file = output_dir / f"{ytid}_{start}-{end}.wav"

    if output_file.exists():
        return True, "Already exists"

    url = f"https://www.youtube.com/watch?v={ytid}"

    # yt-dlp command
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ss {start} -t {end - start}",
        "-o", str(output_file.with_suffix('.%(ext)s')),
        "--quiet",
        "--no-warnings",
        url
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=30, check=True)

        # Check if file was created
        if output_file.exists():
            return True, "Downloaded"
        else:
            return False, "File not created"

    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.returncode}"
    except Exception as e:
        return False, str(e)


def download_label(label_name, label_id, max_samples):
    """특정 라벨의 샘플 다운로드"""
    print(f"\n{'='*80}")
    print(f"📥 {label_name} 다운로드 (최대 {max_samples}개)")
    print(f"{'='*80}")

    # Filter samples with this label
    mask = df['positive_labels'].str.contains(label_id, regex=False, na=False)
    label_df = df[mask].head(max_samples * 3)  # 3x buffer for failures

    print(f"   후보: {len(label_df)}개")

    # Output directory
    output_dir = output_root / label_name.replace(' ', '_').replace('(', '').replace(')', '')
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0

    # Download with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        for idx, row in label_df.iterrows():
            if success_count >= max_samples:
                break

            future = executor.submit(
                download_segment,
                row['YTID'],
                row['start_seconds'],
                row['end_seconds'],
                label_name,
                output_dir
            )
            futures[future] = row['YTID']

        for future in as_completed(futures):
            ytid = futures[future]

            try:
                success, msg = future.result()

                if success:
                    success_count += 1
                    if success_count % 10 == 0:
                        print(f"   [{success_count}/{max_samples}] 다운로드 완료...")
                else:
                    failed_count += 1

                if success_count >= max_samples:
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break

            except Exception as e:
                failed_count += 1

    print(f"\n✅ {label_name} 완료:")
    print(f"   성공: {success_count}개")
    print(f"   실패: {failed_count}개")

    return success_count


# Download each label
results = {}

for label_name, label_id in LABELS.items():
    count = download_label(label_name, label_id, MAX_PER_LABEL)
    results[label_name] = count


# Reorganize files
print("\n" + "="*80)
print("📁 파일 정리 중...")
print("="*80)

normal_dir = Path("data/training/normal")
knocking_dir = Path("data/training/audioset_knocking")

normal_dir.mkdir(parents=True, exist_ok=True)
knocking_dir.mkdir(parents=True, exist_ok=True)

# Move Idling to normal
idling_dir = output_root / "Idling"
if idling_dir.exists():
    for wav_file in idling_dir.glob("*.wav"):
        shutil.copy(wav_file, normal_dir / f"idling_{wav_file.name}")
    print(f"✅ Idling: {len(list(idling_dir.glob('*.wav')))}개 → {normal_dir}")

# Move Medium engine to normal
medium_dir = output_root / "Medium_engine_mid_frequency"
if medium_dir.exists():
    for wav_file in medium_dir.glob("*.wav"):
        shutil.copy(wav_file, normal_dir / f"medium_{wav_file.name}")
    print(f"✅ Medium engine: {len(list(medium_dir.glob('*.wav')))}개 → {normal_dir}")

# Move Engine knocking
knocking_audioset_dir = output_root / "Engine_knocking"
if knocking_audioset_dir.exists():
    for wav_file in knocking_audioset_dir.glob("*.wav"):
        shutil.copy(wav_file, knocking_dir / wav_file.name)
    print(f"✅ Engine knocking: {len(list(knocking_audioset_dir.glob('*.wav')))}개 → {knocking_dir}")


# Summary
print("\n" + "="*80)
print("✅ 다운로드 완료!")
print("="*80)

total_normal = len(list(normal_dir.glob("*.wav")))
total_knocking = len(list(knocking_dir.glob("*.wav")))

print(f"📊 최종 결과:")
print(f"   정상 엔진 소리: {total_normal}개 → {normal_dir}")
print(f"   노킹 소리: {total_knocking}개 → {knocking_dir}")
print(f"\n💡 다음 단계:")
print(f"   python train_two_class.py  # Two-Class 모델 학습")
print("="*80)
