#!/usr/bin/env python3
"""
정상 엔진 소리 YouTube 다운로드
youtube_links_normal.txt에서 링크를 읽어서 다운로드
"""

import subprocess
from pathlib import Path
import re

print("="*80)
print("🎵 정상 엔진 소리 YouTube 다운로드")
print("="*80)

# 입력/출력
links_file = Path("youtube_links_normal.txt")
output_dir = Path("data/training/youtube_normal")
output_dir.mkdir(parents=True, exist_ok=True)

# yt-dlp 설치 확인
try:
    subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    print("✅ yt-dlp 설치 확인됨")
except:
    print("❌ yt-dlp가 설치되지 않았습니다.")
    print("   설치: pip install yt-dlp")
    exit(1)

# 링크 읽기
if not links_file.exists():
    print(f"❌ {links_file}를 찾을 수 없습니다.")
    exit(1)

links = []
with open(links_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            links.append(line)

print(f"\n📂 다운로드할 링크: {len(links)}개")

# 다운로드
success_count = 0
fail_count = 0

for i, url in enumerate(links, 1):
    print(f"\n[{i}/{len(links)}] {url}")

    try:
        # yt-dlp로 오디오만 다운로드 (16kHz, mono, wav)
        subprocess.run([
            "yt-dlp",
            "-x",  # Extract audio
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--postprocessor-args", "-ar 16000 -ac 1",  # 16kHz, mono
            "-o", str(output_dir / "%(id)s.%(ext)s"),  # 비디오 ID로 저장
            url
        ], check=True, capture_output=True, text=True)

        print(f"  ✅ 다운로드 성공")
        success_count += 1

    except subprocess.CalledProcessError as e:
        print(f"  ❌ 다운로드 실패: {e}")
        fail_count += 1
        continue

# 결과
print("\n" + "="*80)
print("📊 다운로드 완료!")
print("="*80)
print(f"   성공: {success_count}개")
print(f"   실패: {fail_count}개")
print(f"   저장 위치: {output_dir}")

# 다운로드된 파일 목록
wav_files = list(output_dir.glob("*.wav"))
print(f"\n📁 다운로드된 파일: {len(wav_files)}개")
for wav_file in wav_files:
    print(f"   - {wav_file.name}")

print(f"\n💡 다음 단계:")
print(f"   1. 다운로드된 파일 확인")
print(f"   2. python augment_normal_sounds.py 수정 (youtube_normal 경로 추가)")
print(f"   3. 정상 데이터 증강 재실행")
print("="*80)
