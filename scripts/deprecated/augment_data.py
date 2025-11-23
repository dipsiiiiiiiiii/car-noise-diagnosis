#!/usr/bin/env python3
"""
Data augmentation for car noise audio samples

Applies realistic augmentation techniques suitable for car noise diagnosis:
1. Background noise addition (road noise, wind)
2. Volume adjustment (different recording distances)
3. Time stretching (RPM variation - engine sounds only)
4. Pitch shifting (different engine sizes - use sparingly)

Usage:
    python augment_data.py
    python augment_data.py --input data/training --output data/training_augmented
    python augment_data.py --factor 4  # 4x augmentation (1 → 4 samples)
"""

import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import random


def add_background_noise(audio: np.ndarray, sample_rate: int, noise_level: float = 0.005) -> np.ndarray:
    """Add realistic background noise (road, wind, ambient)"""

    # Generate colored noise (pink noise - more realistic for car environment)
    # Pink noise has 1/f spectrum (more low-frequency energy)
    noise = np.random.randn(len(audio))

    # Apply low-pass filter to create pink-ish noise
    from scipy import signal
    b, a = signal.butter(2, 0.5)
    noise = signal.filtfilt(b, a, noise)

    # Normalize and scale
    noise = noise / np.max(np.abs(noise))
    noise = noise * noise_level

    # Mix with original
    augmented = audio + noise

    # Prevent clipping
    if np.max(np.abs(augmented)) > 1.0:
        augmented = augmented / np.max(np.abs(augmented)) * 0.95

    return augmented.astype(np.float32)


def change_volume(audio: np.ndarray, volume_factor: float) -> np.ndarray:
    """Adjust volume (simulate different recording distances)"""
    augmented = audio * volume_factor

    # Prevent clipping
    if np.max(np.abs(augmented)) > 1.0:
        augmented = augmented / np.max(np.abs(augmented)) * 0.95

    return augmented.astype(np.float32)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """Time stretching (simulate RPM changes)"""
    augmented = librosa.effects.time_stretch(audio, rate=rate)
    return augmented.astype(np.float32)


def pitch_shift(audio: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    """Pitch shifting (simulate different engine sizes)"""
    augmented = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
    return augmented.astype(np.float32)


def augment_audio(audio: np.ndarray, sample_rate: int,
                  class_name: str, aug_type: str) -> Tuple[np.ndarray, str]:
    """Apply specific augmentation based on type

    Returns:
        augmented_audio, description
    """

    if aug_type == 'noise':
        # Background noise
        noise_level = random.uniform(0.003, 0.008)
        augmented = add_background_noise(audio, sample_rate, noise_level)
        desc = f"noise_{noise_level:.4f}"

    elif aug_type == 'volume':
        # Volume change
        volume_factor = random.uniform(0.7, 1.3)
        augmented = change_volume(audio, volume_factor)
        desc = f"vol_{volume_factor:.2f}"

    elif aug_type == 'stretch':
        # Time stretch (only for engine-related sounds)
        if 'engine' in class_name.lower() or 'normal' in class_name.lower():
            rate = random.uniform(0.9, 1.1)
            augmented = time_stretch(audio, rate)
            desc = f"stretch_{rate:.2f}"
        else:
            # Skip for non-engine sounds
            return None, None

    elif aug_type == 'pitch':
        # Pitch shift (use sparingly, only for engine sounds)
        if 'engine' in class_name.lower():
            n_steps = random.uniform(-1, 1)
            augmented = pitch_shift(audio, sample_rate, n_steps)
            desc = f"pitch_{n_steps:+.1f}"
        else:
            # Skip for non-engine sounds
            return None, None

    elif aug_type == 'combo':
        # Combination: noise + volume
        noise_level = random.uniform(0.003, 0.008)
        volume_factor = random.uniform(0.7, 1.3)
        augmented = add_background_noise(audio, sample_rate, noise_level)
        augmented = change_volume(augmented, volume_factor)
        desc = f"combo_n{noise_level:.4f}_v{volume_factor:.2f}"

    else:
        return None, None

    return augmented, desc


def augment_dataset(input_dir: Path, output_dir: Path,
                    augmentation_factor: int = 4):
    """Augment entire dataset

    Args:
        input_dir: Input directory (e.g., data/training)
        output_dir: Output directory for augmented data
        augmentation_factor: Total samples = original × factor
    """

    print(f"🔧 데이터 증강 시작")
    print("="*70)
    print(f"입력 폴더: {input_dir}")
    print(f"출력 폴더: {output_dir}")
    print(f"증강 배수: {augmentation_factor}x")
    print("="*70)

    # Audio extensions
    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a', '*.flac', '*.ogg']

    # Find all class folders
    class_folders = [d for d in input_dir.iterdir() if d.is_dir()]

    if not class_folders:
        print(f"❌ {input_dir}에 클래스 폴더가 없습니다.")
        return

    total_original = 0
    total_augmented = 0

    for class_folder in sorted(class_folders):
        class_name = class_folder.name
        print(f"\n📁 클래스: {class_name}")
        print("-"*70)

        # Create output folder
        output_class_folder = output_dir / class_name
        output_class_folder.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(class_folder.glob(ext))

        if not audio_files:
            print(f"   ⚠️  오디오 파일 없음")
            continue

        print(f"   원본 파일: {len(audio_files)}개")
        total_original += len(audio_files)

        class_augmented = 0

        for audio_file in audio_files:
            try:
                # Load audio
                audio, sr = librosa.load(str(audio_file), sr=16000)

                # Copy original to output
                output_original = output_class_folder / audio_file.name
                sf.write(str(output_original), audio, sr)

                # Determine augmentation types
                # For each original, create (augmentation_factor - 1) augmented samples
                num_augmentations = augmentation_factor - 1

                # Augmentation strategy
                aug_types = []

                # Always include these
                aug_types.append('noise')
                aug_types.append('volume')
                aug_types.append('combo')

                # Add more based on factor
                if num_augmentations >= 4:
                    aug_types.append('stretch')
                if num_augmentations >= 5:
                    aug_types.append('pitch')

                # If we need more, repeat some
                while len(aug_types) < num_augmentations:
                    aug_types.append(random.choice(['noise', 'volume', 'combo']))

                # Trim to exact number needed
                aug_types = aug_types[:num_augmentations]

                # Apply augmentations
                for i, aug_type in enumerate(aug_types, 1):
                    augmented, desc = augment_audio(audio, sr, class_name, aug_type)

                    if augmented is None:
                        continue

                    # Save augmented audio
                    stem = audio_file.stem
                    output_filename = f"{stem}_aug_{desc}.wav"
                    output_path = output_class_folder / output_filename

                    sf.write(str(output_path), augmented, sr)
                    class_augmented += 1

            except Exception as e:
                print(f"   ⚠️  {audio_file.name} 처리 실패: {e}")
                continue

        total_augmented += class_augmented
        print(f"   증강 파일: {class_augmented}개")
        print(f"   총 파일: {len(audio_files) + class_augmented}개")

    print("\n" + "="*70)
    print(f"✅ 증강 완료")
    print(f"   원본: {total_original}개")
    print(f"   증강: {total_augmented}개")
    print(f"   총: {total_original + total_augmented}개")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Augment car noise audio dataset")
    parser.add_argument('--input', type=str, default='data/training',
                        help='Input directory with class folders')
    parser.add_argument('--output', type=str, default='data/training_augmented',
                        help='Output directory for augmented data')
    parser.add_argument('--factor', type=int, default=4,
                        help='Augmentation factor (default: 4x)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"❌ 입력 폴더가 없습니다: {input_dir}")
        return 1

    # Run augmentation
    augment_dataset(input_dir, output_dir, args.factor)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
