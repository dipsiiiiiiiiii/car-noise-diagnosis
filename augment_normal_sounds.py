#!/usr/bin/env python3
"""
정상 엔진 소리 Data Augmentation
34개 → 150개로 증강
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import random

print("="*80)
print("🔄 정상 엔진 소리 Data Augmentation")
print("="*80)

# 입력/출력 디렉토리
input_dirs = [
    Path("data/training/audioset/Idling"),
    Path("data/training/audioset/Medium_engine_mid_frequency")
]
output_dir = Path("data/training/normal")
output_dir.mkdir(parents=True, exist_ok=True)

# 원본 파일 수집
original_files = []
for input_dir in input_dirs:
    if input_dir.exists():
        original_files.extend(list(input_dir.glob("*.wav")))

print(f"\n📂 원본 파일: {len(original_files)}개")
print(f"   - Idling: {len(list(input_dirs[0].glob('*.wav')))}개")
print(f"   - Medium engine: {len(list(input_dirs[1].glob('*.wav')))}개")

# Augmentation 함수들
def time_stretch(audio, rate):
    """시간 늘리기/줄이기"""
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sr, n_steps):
    """피치 변경"""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor):
    """화이트 노이즈 추가"""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def change_volume(audio, factor):
    """볼륨 변경"""
    return audio * factor

def time_shift(audio, shift_max):
    """시간 이동"""
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)


# Augmentation 설정
augmentations = [
    ("original", lambda x, sr: x),
    ("time_stretch_0.9", lambda x, sr: time_stretch(x, 0.9)),
    ("time_stretch_1.1", lambda x, sr: time_stretch(x, 1.1)),
    ("pitch_shift_-2", lambda x, sr: pitch_shift(x, sr, -2)),
    ("pitch_shift_2", lambda x, sr: pitch_shift(x, sr, 2)),
    ("noise_0.005", lambda x, sr: add_noise(x, 0.005)),
    ("volume_0.8", lambda x, sr: change_volume(x, 0.8)),
    ("volume_1.2", lambda x, sr: change_volume(x, 1.2)),
]

target_count = 150
augment_per_file = target_count // len(original_files) + 1

print(f"\n🎯 목표: {target_count}개")
print(f"   파일당 augmentation: {augment_per_file}개")

# Augmentation 수행
total_created = 0

print(f"\n⚙️  Augmentation 진행 중...")
for i, audio_file in enumerate(original_files, 1):
    try:
        # 원본 로드
        audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)

        # 원본 저장
        source = "idling" if "Idling" in str(audio_file) else "medium"
        output_path = output_dir / f"{source}_{audio_file.stem}_original.wav"
        sf.write(output_path, audio, sr)
        total_created += 1

        # Augmentation 적용
        num_augments = min(augment_per_file - 1, len(augmentations) - 1)
        selected_augs = random.sample(augmentations[1:], num_augments)

        for aug_name, aug_func in selected_augs:
            try:
                augmented = aug_func(audio, sr)

                # Normalize
                augmented = np.clip(augmented, -1.0, 1.0)

                # 저장
                output_path = output_dir / f"{source}_{audio_file.stem}_{aug_name}.wav"
                sf.write(output_path, augmented, sr)
                total_created += 1

            except Exception as e:
                print(f"  ⚠️  Aug 실패 ({aug_name}): {audio_file.name} - {e}")
                continue

        if i % 5 == 0 or i == len(original_files):
            print(f"  [{i}/{len(original_files)}] 처리 완료... (생성: {total_created}개)")

    except Exception as e:
        print(f"  ❌ 파일 로드 실패: {audio_file.name} - {e}")
        continue

# 결과
print("\n" + "="*80)
print("✅ Augmentation 완료!")
print("="*80)
print(f"📊 결과:")
print(f"   - 원본: {len(original_files)}개")
print(f"   - 생성됨: {total_created}개")
print(f"   - 최종: {total_created}개 → {output_dir}")
print(f"\n💡 다음 단계:")
print(f"   python train_two_class.py  # Two-Class 모델 학습")
print("="*80)
