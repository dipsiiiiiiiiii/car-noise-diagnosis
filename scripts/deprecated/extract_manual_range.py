#!/usr/bin/env python3
"""
Manual Time Range Extraction
사용자가 직접 시간 범위를 지정해서 노킹 구간 추출
"""

import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import argparse

sys.path.append(str(Path(__file__).parent / "src"))

from audio.capture import AudioFileLoader


def extract_time_range(audio_path: Path, start_time: float, end_time: float,
                       output_dir: Path, segment_name: str = None):
    """Extract specific time range from audio file"""

    # Load audio
    print(f"\n📂 로딩: {audio_path.name}")
    audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_path))
    duration = len(audio_data) / sample_rate

    print(f"   전체 길이: {duration:.1f}초")
    print(f"   추출 범위: {start_time:.1f}초 ~ {end_time:.1f}초")

    # Validate times
    if start_time < 0 or end_time > duration:
        print(f"   ❌ 시간 범위가 유효하지 않습니다 (0 ~ {duration:.1f}초)")
        return False

    if start_time >= end_time:
        print(f"   ❌ 시작 시간이 끝 시간보다 늦습니다")
        return False

    # Extract segment
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    segment = audio_data[start_sample:end_sample]

    # Generate output name
    if segment_name is None:
        segment_name = f"{audio_path.stem}_t{start_time:.1f}-{end_time:.1f}s.wav"
    elif not segment_name.endswith('.wav'):
        segment_name += '.wav'

    output_path = output_dir / segment_name

    # Save
    sf.write(str(output_path), segment, sample_rate)
    print(f"   ✅ 저장: {output_path.name}")

    # Show RMS
    rms = np.sqrt(np.mean(segment ** 2))
    print(f"   📊 RMS: {rms:.4f}")

    return True


def interactive_mode():
    """Interactive extraction mode"""

    print("="*80)
    print("🎯 수동 시간 범위 추출 (Interactive Mode)")
    print("="*80)

    # Setup paths
    test_dir = Path("data/testing/batch_test")
    output_dir = Path("data/training/manual_workflow/2_verified")

    output_dir.mkdir(parents=True, exist_ok=True)

    # List available videos
    videos = sorted(test_dir.glob("video_*.wav"))

    if not videos:
        print("\n❌ 영상을 찾을 수 없습니다. 먼저 batch_test_videos.py를 실행하세요.")
        return

    print(f"\n📁 사용 가능한 영상들:")
    for v in videos:
        print(f"   {v.stem}")

    print(f"\n💡 사용법:")
    print(f"   - 영상 번호, 시작 시간, 끝 시간을 입력하세요")
    print(f"   - 예: 10 5.0 8.5  (10번 영상의 5.0~8.5초 추출)")
    print(f"   - 'q'를 입력하면 종료")
    print("="*80)

    extracted_count = 0

    while True:
        print(f"\n입력 (영상번호 시작시간 끝시간) 또는 'q': ", end="")
        user_input = input().strip()

        if user_input.lower() == 'q':
            break

        try:
            parts = user_input.split()

            if len(parts) != 3:
                print("❌ 형식: 영상번호 시작시간 끝시간 (예: 10 5.0 8.5)")
                continue

            video_num = int(parts[0])
            start_time = float(parts[1])
            end_time = float(parts[2])

            # Find video file
            video_file = test_dir / f"video_{video_num:02d}.wav"

            if not video_file.exists():
                print(f"❌ video_{video_num:02d}.wav를 찾을 수 없습니다")
                continue

            # Extract
            segment_name = f"manual_v{video_num:02d}_t{start_time:.1f}-{end_time:.1f}s"
            if extract_time_range(video_file, start_time, end_time, output_dir, segment_name):
                extracted_count += 1
                print(f"   🎉 총 {extracted_count}개 추출 완료")

        except ValueError as e:
            print(f"❌ 입력 오류: {e}")
        except Exception as e:
            print(f"❌ 추출 실패: {e}")

    print("\n" + "="*80)
    print(f"✅ 총 {extracted_count}개 세그먼트 추출 완료!")
    print(f"📁 저장 위치: {output_dir}")
    print("="*80)


def batch_mode(video_num: int, ranges: list):
    """Batch extraction mode"""

    print("="*80)
    print("🎯 수동 시간 범위 추출 (Batch Mode)")
    print("="*80)

    test_dir = Path("data/testing/batch_test")
    output_dir = Path("data/training/manual_workflow/2_verified")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_file = test_dir / f"video_{video_num:02d}.wav"

    if not video_file.exists():
        print(f"\n❌ video_{video_num:02d}.wav를 찾을 수 없습니다")
        return

    print(f"\n📂 영상: {video_file.name}")
    print(f"📊 총 {len(ranges)}개 범위 추출")
    print("="*80)

    extracted = 0
    for i, (start, end) in enumerate(ranges, 1):
        print(f"\n[{i}/{len(ranges)}]")
        segment_name = f"manual_v{video_num:02d}_range{i:02d}_t{start:.1f}-{end:.1f}s"
        if extract_time_range(video_file, start, end, output_dir, segment_name):
            extracted += 1

    print("\n" + "="*80)
    print(f"✅ {extracted}/{len(ranges)}개 추출 완료!")
    print(f"📁 저장 위치: {output_dir}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='수동 시간 범위 추출')
    parser.add_argument('-v', '--video', type=int, help='영상 번호')
    parser.add_argument('-r', '--ranges', type=str, help='시간 범위 (예: "5-8,10-15,20-25")')

    args = parser.parse_args()

    if args.video and args.ranges:
        # Batch mode
        try:
            ranges = []
            for r in args.ranges.split(','):
                start, end = map(float, r.split('-'))
                ranges.append((start, end))
            batch_mode(args.video, ranges)
        except Exception as e:
            print(f"❌ 범위 파싱 오류: {e}")
            print("형식: -r \"5-8,10-15,20-25\"")
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
