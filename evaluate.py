#!/usr/bin/env python3
"""
Evaluate and compare YAMNet baseline vs custom trained model

Generates comparison graphs for presentation:
- Accuracy comparison
- Confusion matrices
- Per-class performance metrics
- Confidence distribution

Usage:
    python evaluate.py
    python evaluate.py --data-dir data/training --model data/models/car_classifier.pkl
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pickle
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.mediapipe_classifier import MediaPipeAudioClassifier
from audio.capture import AudioFileLoader
from diagnosis.analyzer import (
    CarNoiseDiagnoser, PROBLEM_LABELS, LABEL_TO_NAME
)


def load_test_data(data_dir: Path, classifier: MediaPipeAudioClassifier,
                   test_ratio: float = 0.2) -> Tuple[List, List, List, List]:
    """Load test data

    Args:
        data_dir: Training data directory
        classifier: MediaPipe classifier
        test_ratio: Ratio of data to use for testing

    Returns:
        audio_data_list: List of audio arrays
        embeddings_list: List of embeddings
        labels_list: List of labels
        mediapipe_results_list: List of YAMNet results
    """
    audio_data_list = []
    embeddings_list = []
    labels_list = []
    mediapipe_results_list = []
    file_paths = []

    print(f"\n📂 테스트 데이터 로딩: {data_dir}")
    print("=" * 60)

    audio_extensions = ['*.wav', '*.mp3', '*.mp4', '*.m4a', '*.flac', '*.ogg']

    np.random.seed(42)  # For reproducible test split

    for label_name, label_id in PROBLEM_LABELS.items():
        class_dir = data_dir / label_name

        if not class_dir.exists():
            continue

        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(class_dir.glob(ext))

        if not audio_files:
            continue

        # Take subset as test data
        n_test = max(1, int(len(audio_files) * test_ratio))
        test_files = np.random.choice(audio_files, size=n_test, replace=False)

        print(f"\n🔊 {label_name} ({label_id}): {len(test_files)}개 테스트 파일")

        for i, audio_file in enumerate(test_files, 1):
            try:
                # Load audio
                audio_data, sample_rate = AudioFileLoader.load_audio(str(audio_file))

                if len(audio_data) == 0:
                    continue

                # Get YAMNet results
                mediapipe_results = classifier.classify_audio(audio_data, sample_rate)

                # Extract embedding
                embedding = classifier.extract_embedding(audio_data, sample_rate)

                if embedding is None or not mediapipe_results:
                    continue

                audio_data_list.append(audio_data)
                embeddings_list.append(embedding)
                labels_list.append(label_id)
                mediapipe_results_list.append(mediapipe_results)
                file_paths.append(str(audio_file))

                print(f"  ✅ [{i}/{len(test_files)}] {audio_file.name}")

            except Exception as e:
                print(f"  ❌ [{i}/{len(test_files)}] {audio_file.name}: {e}")
                continue

    print("\n" + "=" * 60)
    print(f"✅ 총 {len(embeddings_list)}개 테스트 샘플")

    return audio_data_list, embeddings_list, labels_list, mediapipe_results_list


def evaluate_baseline(mediapipe_results_list: List, labels: List) -> Tuple[List, List, List]:
    """Evaluate baseline using YAMNet only

    Returns:
        predictions, confidences, issues_found
    """
    print("\n🔍 Baseline (YAMNet만) 평가 중...")
    diagnoser = CarNoiseDiagnoser(model_path=None)  # Baseline mode

    predictions = []
    confidences = []
    issues_found = []

    for mediapipe_results in mediapipe_results_list:
        # Use baseline diagnosis
        diagnosis = diagnoser._baseline_diagnose(mediapipe_results)

        # Map to label (simple heuristic)
        if diagnosis['issues']:
            # Has issues - try to map to specific problem
            first_issue = diagnosis['issues'][0]
            part = first_issue['part']

            # Map part to label (simplified)
            if "브레이크" in part:
                pred = PROBLEM_LABELS['brake_problem']
            elif "엔진" in part:
                pred = PROBLEM_LABELS['engine_problem']
            elif "베어링" in part:
                pred = PROBLEM_LABELS['bearing_problem']
            elif "벨트" in part:
                pred = PROBLEM_LABELS['belt_problem']
            elif "타이어" in part:
                pred = PROBLEM_LABELS['tire_problem']
            elif "변속기" in part:
                pred = PROBLEM_LABELS['transmission_problem']
            else:
                pred = PROBLEM_LABELS['normal']  # Unknown -> normal

            conf = diagnosis['confidence']
        else:
            # No issues = normal
            pred = PROBLEM_LABELS['normal']
            conf = diagnosis['confidence']

        predictions.append(pred)
        confidences.append(conf)
        issues_found.append(len(diagnosis['issues']))

    return predictions, confidences, issues_found


def evaluate_custom(embeddings: List[np.ndarray], model_path: Path) -> Tuple[List, List]:
    """Evaluate custom trained model

    Returns:
        predictions, confidences
    """
    print("\n🎯 Custom Model 평가 중...")

    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    # Load model
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)

    predictions = []
    confidences = []

    X = np.array(embeddings)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    for pred, prob in zip(preds, probs):
        predictions.append(int(pred))
        confidences.append(float(prob[pred]))

    return predictions, confidences


def generate_comparison_graphs(
    labels: List[int],
    baseline_preds: List[int],
    baseline_confs: List[float],
    custom_preds: List[int],
    custom_confs: List[float],
    output_dir: Path
):
    """Generate comparison graphs for presentation"""

    print("\n📊 비교 그래프 생성 중...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = ['AppleGothic', 'sans-serif']  # Korean font
    plt.rcParams['axes.unicode_minus'] = False

    # Convert to numpy
    labels = np.array(labels)
    baseline_preds = np.array(baseline_preds)
    custom_preds = np.array(custom_preds)

    class_names = [LABEL_TO_NAME[i] for i in sorted(PROBLEM_LABELS.values())]

    # 1. Accuracy Comparison
    print("  1/4 정확도 비교 차트...")
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_acc = accuracy_score(labels, baseline_preds)
    custom_acc = accuracy_score(labels, custom_preds)

    models = ['YAMNet\nBaseline', 'Custom Model\n(Trained)']
    accuracies = [baseline_acc * 100, custom_acc * 100]
    colors = ['#FF6B6B', '#4ECDC4']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"    ✅ 저장: {output_dir / 'accuracy_comparison.png'}")
    plt.close()

    # 2. Confusion Matrices
    print("  2/4 혼동 행렬...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Baseline confusion matrix
    cm_baseline = confusion_matrix(labels, baseline_preds)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    axes[0].set_title(f'YAMNet Baseline\n(Accuracy: {baseline_acc:.1%})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    # Custom model confusion matrix
    cm_custom = confusion_matrix(labels, custom_preds)
    sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    axes[1].set_title(f'Custom Model\n(Accuracy: {custom_acc:.1%})', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"    ✅ 저장: {output_dir / 'confusion_matrices.png'}")
    plt.close()

    # 3. Per-class Performance
    print("  3/4 클래스별 성능...")
    baseline_metrics = precision_recall_fscore_support(labels, baseline_preds, average=None, zero_division=0)
    custom_metrics = precision_recall_fscore_support(labels, custom_preds, average=None, zero_division=0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(class_names))
    width = 0.35

    for idx, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        baseline_vals = baseline_metrics[idx] * 100
        custom_vals = custom_metrics[idx] * 100

        ax.bar(x - width/2, baseline_vals, width, label='YAMNet Baseline', color='#FF6B6B', alpha=0.8)
        ax.bar(x + width/2, custom_vals, width, label='Custom Model', color='#4ECDC4', alpha=0.8)

        ax.set_ylabel(f'{metric_name} (%)', fontsize=12)
        ax.set_title(f'{metric_name} by Class', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    print(f"    ✅ 저장: {output_dir / 'per_class_performance.png'}")
    plt.close()

    # 4. Confidence Distribution
    print("  4/4 신뢰도 분포...")
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(baseline_confs, bins=20, alpha=0.6, label='YAMNet Baseline', color='#FF6B6B', edgecolor='black')
    ax.hist(custom_confs, bins=20, alpha=0.6, label='Custom Model', color='#4ECDC4', edgecolor='black')

    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add mean lines
    ax.axvline(np.mean(baseline_confs), color='#FF6B6B', linestyle='--', linewidth=2,
               label=f'Baseline Mean: {np.mean(baseline_confs):.2f}')
    ax.axvline(np.mean(custom_confs), color='#4ECDC4', linestyle='--', linewidth=2,
               label=f'Custom Mean: {np.mean(custom_confs):.2f}')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"    ✅ 저장: {output_dir / 'confidence_distribution.png'}")
    plt.close()

    print("\n✅ 모든 그래프 생성 완료!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument('--data-dir', type=str, default='data/training')
    parser.add_argument('--model', type=str, default='data/models/car_classifier.pkl')
    parser.add_argument('--yamnet', type=str, default='data/models/yamnet.tflite')
    parser.add_argument('--output', type=str, default='data/evaluation_results')
    parser.add_argument('--test-ratio', type=float, default=0.2)

    args = parser.parse_args()

    print("📊 모델 평가 및 비교")
    print("=" * 60)

    data_dir = Path(args.data_dir)
    model_path = Path(args.model)
    yamnet_path = Path(args.yamnet)
    output_dir = Path(args.output)

    # Check paths
    if not data_dir.exists():
        print(f"❌ 데이터 디렉토리 없음: {data_dir}")
        return 1

    if not model_path.exists():
        print(f"❌ 학습된 모델 없음: {model_path}")
        print("먼저 train.py를 실행하세요.")
        return 1

    if not yamnet_path.exists():
        print(f"❌ YAMNet 모델 없음: {yamnet_path}")
        return 1

    # Initialize classifier
    print(f"\n🤖 YAMNet 로딩: {yamnet_path}")
    classifier = MediaPipeAudioClassifier(
        model_path=str(yamnet_path),
        max_results=50,
        score_threshold=0.0
    )

    # Load test data
    audio_list, embeddings, labels, mediapipe_results = load_test_data(
        data_dir, classifier, args.test_ratio
    )

    if len(labels) < 5:
        print(f"❌ 테스트 데이터가 너무 적습니다 ({len(labels)}개)")
        return 1

    # Evaluate baseline
    baseline_preds, baseline_confs, _ = evaluate_baseline(mediapipe_results, labels)

    # Evaluate custom model
    custom_preds, custom_confs = evaluate_custom(embeddings, model_path)

    # Print results
    print("\n" + "=" * 60)
    print("📈 평가 결과:")
    print("=" * 60)

    baseline_acc = accuracy_score(labels, baseline_preds)
    custom_acc = accuracy_score(labels, custom_preds)

    print(f"\n1️⃣  YAMNet Baseline:")
    print(f"   Accuracy: {baseline_acc:.2%}")
    print(f"   Mean Confidence: {np.mean(baseline_confs):.2f}")

    print(f"\n2️⃣  Custom Trained Model:")
    print(f"   Accuracy: {custom_acc:.2%}")
    print(f"   Mean Confidence: {np.mean(custom_confs):.2f}")

    print(f"\n✨ 개선율: {(custom_acc - baseline_acc) / baseline_acc * 100:+.1f}%")

    # Generate graphs
    generate_comparison_graphs(
        labels, baseline_preds, baseline_confs,
        custom_preds, custom_confs, output_dir
    )

    print(f"\n💾 결과 저장 위치: {output_dir}")
    print("\n발표에 사용할 그래프:")
    print(f"  - {output_dir / 'accuracy_comparison.png'}")
    print(f"  - {output_dir / 'confusion_matrices.png'}")
    print(f"  - {output_dir / 'per_class_performance.png'}")
    print(f"  - {output_dir / 'confidence_distribution.png'}")

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
