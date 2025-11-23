#!/usr/bin/env python3
"""
노킹 소리 Data Augmentation
172개 → 240개로 증강 (정상 데이터와 균형 맞추기)
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import random

print("="*80)
print("🔄 노킹 소리 Data Augmentation")
print("="*80)

# 입력/출력 디렉토리
input_dir = Path("data/training/manual_workflow/2_verified")
output_dir = Path("data/training/knocking_augmented")
output_dir.mkdir(parents=True, exist_ok=True)

# 원본 파일 수집
original_files = []
if input_dir.exists():
    original_files = list(input_dir.glob("*.wav"))

print(f"\n📂 원본 파일: {len(original_files)}개")
print(f"   - Verified knocking: {len(original_files)}개")

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


# Augmentation 설정 (노킹 특성 유지를 위해 약한 변형)
augmentations = [
    ("original", lambda x, sr: x),
    ("time_stretch_0.95", lambda x, sr: time_stretch(x, 0.95)),  # 더 미세한 변화
    ("time_stretch_1.05", lambda x, sr: time_stretch(x, 1.05)),
    ("pitch_shift_-1", lambda x, sr: pitch_shift(x, sr, -1)),  # 더 미세한 피치 변화
    ("pitch_shift_1", lambda x, sr: pitch_shift(x, sr, 1)),
    ("noise_0.003", lambda x, sr: add_noise(x, 0.003)),  # 더 약한 노이즈
    ("volume_0.9", lambda x, sr: change_volume(x, 0.9)),
    ("volume_1.1", lambda x, sr: change_volume(x, 1.1)),
]

target_count = 240
# 원본 포함해서 파일당 몇 개를 만들지 계산 (올림)
augment_per_file = ((target_count + len(original_files) - 1) // len(original_files)) if len(original_files) > 0 else 0

print(f"\n🎯 목표: {target_count}개")
print(f"   파일당 총 생성 개수: {augment_per_file}개 (원본 1개 + 증강 {augment_per_file-1}개)")

# Augmentation 수행
total_created = 0

print(f"\n⚙️  Augmentation 진행 중...")
for i, audio_file in enumerate(original_files, 1):
    try:
        # 원본 로드
        audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)

        # 원본 저장
        output_path = output_dir / f"knocking_{audio_file.stem}_original.wav"
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
                output_path = output_dir / f"knocking_{audio_file.stem}_{aug_name}.wav"
                sf.write(output_path, augmented, sr)
                total_created += 1

            except Exception as e:
                print(f"  ⚠️  Aug 실패 ({aug_name}): {audio_file.name} - {e}")
                continue

        if i % 20 == 0 or i == len(original_files):
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

# 전체 데이터 요약
normal_count = len(list(Path("data/training/normal").glob("*.wav"))) if Path("data/training/normal").exists() else 0
print(f"\n📈 전체 데이터셋 균형:")
print(f"   - 정상: {normal_count}개")
print(f"   - 노킹: {total_created}개")
print(f"   - 비율: 정상 {normal_count/(normal_count+total_created)*100:.1f}% / 노킹 {total_created/(normal_count+total_created)*100:.1f}%")

print(f"\n💡 다음 단계:")
print(f"   1. train_two_class.py에서 'data/training/knocking_augmented' 경로 추가")
print(f"   2. python train_two_class.py  # 모델 재학습")
print("="*80)
