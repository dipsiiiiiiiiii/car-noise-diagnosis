#!/usr/bin/env python3
"""
정상 엔진 소리 수동 구간 추출
YouTube에서 다운로드한 영상을 듣고 정상 구간을 수동으로 추출
"""

import sys
import numpy as np
from pathlib import Path
import soundfile as sf
import subprocess

sys.path.append(str(Path(__file__).parent / "src"))

from audio.capture import AudioFileLoader


def play_audio(audio_path: Path):
    """Play audio file using afplay (macOS)"""
    try:
        subprocess.run(['afplay', str(audio_path)], check=True)
    except FileNotFoundError:
        print("  ⚠️  afplay를 찾을 수 없습니다. macOS에서만 작동합니다.")
    except Exception as e:
        print(f"  ⚠️  재생 오류: {e}")


def extract_time_range(audio_path: Path, start_time: float, end_time: float,
                       output_dir: Path, segment_name: str = None):
    """Extract specific time range from audio file"""

    # Load audio
    audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_path))
    duration = len(audio_data) / sample_rate

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
        segment_name = f"{audio_path.stem}_{start_time:.1f}-{end_time:.1f}s.wav"
    elif not segment_name.endswith('.wav'):
        segment_name += '.wav'

    output_path = output_dir / segment_name

    # Save
    sf.write(str(output_path), segment, sample_rate)
    print(f"   ✅ 저장: {output_path.name}")

    # Show RMS
    rms = np.sqrt(np.mean(segment ** 2))
    print(f"   📊 RMS: {rms:.4f}, 길이: {end_time - start_time:.1f}초")

    return True


def interactive_mode():
    """Interactive extraction mode"""

    print("="*80)
    print("🎯 정상 엔진 소리 수동 구간 추출")
    print("="*80)

    # Setup paths
    input_dir = Path("data/training/raw/youtube/normal")
    output_dir = Path("data/training/manual_review/normal/2_verified")

    if not input_dir.exists():
        print(f"\n❌ 입력 폴더를 찾을 수 없습니다: {input_dir}")
        print("먼저 download_normal_youtube.py를 실행하세요.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # List available audio files
    audio_files = sorted(input_dir.glob("*.wav"))

    if not audio_files:
        print(f"\n❌ {input_dir}에 오디오 파일이 없습니다")
        return

    print(f"\n📁 사용 가능한 파일들:")
    for i, audio_file in enumerate(audio_files, 1):
        audio_data, sr = AudioFileLoader.load_audio(str(audio_file))
        duration = len(audio_data) / sr
        print(f"   {i}. {audio_file.name} (길이: {duration:.1f}초)")

    print(f"\n💡 사용법:")
    print(f"   p <번호>              - 파일 전체 재생")
    print(f"   e <번호> <시작> <끝>  - 구간 추출 (예: e 1 5.0 10.0)")
    print(f"   q                     - 종료")
    print("="*80)

    extracted_count = 0

    while True:
        print(f"\n명령 입력: ", end="")
        user_input = input().strip()

        if user_input.lower() == 'q':
            break

        try:
            parts = user_input.split()

            if len(parts) == 0:
                continue

            command = parts[0].lower()

            if command == 'p':
                # Play file
                if len(parts) != 2:
                    print("❌ 형식: p <파일번호>")
                    continue

                file_num = int(parts[1])
                if file_num < 1 or file_num > len(audio_files):
                    print(f"❌ 파일 번호는 1~{len(audio_files)} 사이여야 합니다")
                    continue

                audio_file = audio_files[file_num - 1]
                print(f"🔊 재생 중: {audio_file.name}")
                play_audio(audio_file)

            elif command == 'e':
                # Extract segment
                if len(parts) != 4:
                    print("❌ 형식: e <파일번호> <시작시간> <끝시간>")
                    continue

                file_num = int(parts[1])
                start_time = float(parts[2])
                end_time = float(parts[3])

                if file_num < 1 or file_num > len(audio_files):
                    print(f"❌ 파일 번호는 1~{len(audio_files)} 사이여야 합니다")
                    continue

                audio_file = audio_files[file_num - 1]
                print(f"\n📂 추출: {audio_file.name}")
                print(f"   범위: {start_time:.1f}초 ~ {end_time:.1f}초")

                # Extract
                segment_name = f"normal_{audio_file.stem}_{start_time:.1f}-{end_time:.1f}s"
                if extract_time_range(audio_file, start_time, end_time, output_dir, segment_name):
                    extracted_count += 1
                    print(f"   🎉 총 {extracted_count}개 추출 완료")

            else:
                print("❌ 알 수 없는 명령. p (재생) 또는 e (추출)를 사용하세요")

        except ValueError as e:
            print(f"❌ 입력 오류: {e}")
        except Exception as e:
            print(f"❌ 오류: {e}")

    print("\n" + "="*80)
    print(f"✅ 총 {extracted_count}개 세그먼트 추출 완료!")
    print(f"📁 저장 위치: {output_dir}")
    print("\n💡 다음 단계:")
    print("   1. 증강 스크립트 업데이트")
    print("   2. python train_two_class.py  # 모델 재학습")
    print("="*80)


if __name__ == "__main__":
    interactive_mode()
