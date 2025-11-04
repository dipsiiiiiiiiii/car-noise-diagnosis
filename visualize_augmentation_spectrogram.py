import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def visualize_spectrogram_comparison(input_dir):
    """Visualize spectrograms of original and augmented audio samples."""

    # Find all audio files in the directory
    audio_files = []
    for ext in ['*.wav', '*.mp3']:
        audio_files.extend(Path(input_dir).rglob(ext))

    # Group files by original sample
    samples = {}
    for file in audio_files:
        # Extract base name (before augmentation suffix)
        filename = file.name
        if '_aug_' in filename:
            base_name = filename.split('_aug_')[0]
        else:
            base_name = file.stem

        if base_name not in samples:
            samples[base_name] = {'original': None, 'augmented': []}

        if '_aug_' not in filename:
            samples[base_name]['original'] = file
        else:
            samples[base_name]['augmented'].append(file)

    # Create visualization for each sample group
    for base_name, files in samples.items():
        print(f"\nProcessing {base_name}...")

        # Prepare subplot layout
        num_files = 1 + len(files['augmented'])  # original + augmented versions
        fig, axes = plt.subplots(num_files, 1, figsize=(15, 4 * num_files))

        if num_files == 1:
            axes = [axes]

        # Plot original
        idx = 0
        if files['original']:
            print(f"  Loading original: {files['original'].name}")
            y, sr = librosa.load(files['original'], sr=None)

            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',
                                          sr=sr, fmax=8000, ax=axes[idx], cmap='viridis')
            axes[idx].set_title(f'Original: {files["original"].name}',
                               fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Frequency (Hz)')
            fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')
            idx += 1

        # Plot augmented versions
        for aug_file in sorted(files['augmented']):
            print(f"  Loading augmented: {aug_file.name}")
            y, sr = librosa.load(aug_file, sr=None)

            # Compute mel spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',
                                          sr=sr, fmax=8000, ax=axes[idx], cmap='viridis')

            # Extract augmentation type from filename
            filename = aug_file.name
            if 'combo' in filename:
                aug_type = 'Combo (Noise + Volume)'
            elif 'noise' in filename:
                aug_type = 'Noise Addition'
            elif 'vol' in filename:
                aug_type = 'Volume Change'
            else:
                aug_type = 'Unknown'

            axes[idx].set_title(f'{aug_type}: {aug_file.name}',
                               fontsize=10, fontweight='bold')
            axes[idx].set_ylabel('Frequency (Hz)')
            fig.colorbar(img, ax=axes[idx], format='%+2.0f dB')
            idx += 1

        axes[-1].set_xlabel('Time (s)')

        plt.tight_layout()

        # Save figure
        output_dir = Path(input_dir) / 'spectrograms'
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f'{base_name}_spectrogram_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")

        plt.close()

    print(f"\n✓ All spectrograms saved to {output_dir}")

if __name__ == '__main__':
    input_dir = 'data/test_augmentation_output'

    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
    else:
        print(f"Visualizing spectrograms from {input_dir}...")
        visualize_spectrogram_comparison(input_dir)
