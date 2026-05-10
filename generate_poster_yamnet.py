#!/usr/bin/env python3
"""
YAMNet Transfer Learning 포스터용 그래프 생성
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
output_dir = Path("poster_yamnet")
output_dir.mkdir(exist_ok=True)

# 데이터 (실제 값)
TOTAL_SAMPLES = 698
NORMAL_SAMPLES = 182
KNOCKING_SAMPLES = 516


def generate_system_architecture():
    """시스템 아키텍처"""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.axis('off')

    arrow_props = dict(arrowstyle='->', lw=3, color='black')

    # 단계별 박스
    stages = [
        (0.5, 0.95, '마이크 입력\n(실시간/파일)', '#E3F2FD'),
        (0.5, 0.85, 'YAMNet (TensorFlow Hub)\n범용 음향 인식 모델', '#90CAF9'),
        (0.5, 0.75, '1024차원 임베딩 추출', '#64B5F6'),
        (0.5, 0.65, 'Transfer Learning\nDense Layers (256→128→64→2)', '#42A5F5'),
        (0.5, 0.55, '정상 vs 노킹 분류', '#66BB6A'),
        (0.5, 0.45, '진단 결과 출력', '#4CAF50'),
    ]

    for i, (x, y, text, color) in enumerate(stages):
        bbox = dict(boxstyle='round,pad=0.7', facecolor=color,
                   edgecolor='black', linewidth=3, alpha=0.9)
        ax.text(x, y, text, ha='center', va='center', fontsize=13,
                fontweight='bold', bbox=bbox, transform=ax.transAxes)

        if i < len(stages) - 1:
            ax.annotate('', xy=(x, stages[i+1][1] + 0.04),
                       xytext=(x, y - 0.04),
                       arrowprops=arrow_props, transform=ax.transAxes)

    # 제목
    ax.text(0.5, 0.99, 'YAMNet Transfer Learning 시스템', ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes)

    # 핵심 정보
    info_text = (
        "📊 핵심 구성\n\n"
        "• YAMNet: Google 공식 모델 (freeze)\n"
        "• 임베딩: 1024차원 벡터\n"
        "• 학습 데이터: 698개 샘플\n"
        "• 커스텀 레이어: Dense (256-128-64)\n"
        "• 배포: 라즈베리파이 5\n"
        "• 실시간/파일 분석 지원"
    )
    bbox_info = dict(boxstyle='round,pad=1', facecolor='#FFF9C4',
                    edgecolor='black', linewidth=2, alpha=0.95)
    ax.text(0.5, 0.22, info_text, ha='center', va='center',
            fontsize=11, bbox=bbox_info, transform=ax.transAxes,
            linespacing=1.8)

    plt.tight_layout()
    plt.savefig(output_dir / '1_system_architecture.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 1_system_architecture.png")
    plt.close()


def generate_model_structure():
    """모델 구조도"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')

    # 레이어 정보
    layers = [
        ('Input\n(1024D)', 0.9, '#E3F2FD', '1024'),
        ('Dense 256\n+ ReLU + BN + Dropout', 0.75, '#90CAF9', '256'),
        ('Dense 128\n+ ReLU + BN + Dropout', 0.60, '#64B5F6', '128'),
        ('Dense 64\n+ ReLU + Dropout', 0.45, '#42A5F5', '64'),
        ('Output\n(2 classes: 정상/노킹)', 0.30, '#4CAF50', '2'),
    ]

    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')

    for i, (name, y_pos, color, size) in enumerate(layers):
        # 레이어 박스
        bbox = dict(boxstyle='round,pad=0.8', facecolor=color,
                   edgecolor='black', linewidth=2.5, alpha=0.9)
        ax.text(0.5, y_pos, name, ha='center', va='center',
                fontsize=13, fontweight='bold', bbox=bbox,
                transform=ax.transAxes)

        # 크기 표시
        ax.text(0.85, y_pos, f'[{size}]', ha='left', va='center',
                fontsize=11, fontweight='bold', transform=ax.transAxes,
                color='darkblue')

        # 화살표
        if i < len(layers) - 1:
            ax.annotate('', xy=(0.5, layers[i+1][1] + 0.06),
                       xytext=(0.5, y_pos - 0.06),
                       arrowprops=arrow_props, transform=ax.transAxes)

    # 제목
    ax.text(0.5, 0.98, 'Transfer Learning 모델 구조', ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes)

    # YAMNet frozen 표시
    bbox_frozen = dict(boxstyle='round,pad=0.5', facecolor='#FFE0B2',
                      edgecolor='red', linewidth=2, alpha=0.9)
    ax.text(0.5, 0.15, '⚠️ YAMNet은 Freeze (학습 안 함)\n커스텀 Dense Layer만 학습',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=bbox_frozen, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '2_model_structure.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 2_model_structure.png")
    plt.close()


def generate_dataset_distribution():
    """데이터셋 분포"""
    fig, ax = plt.subplots(figsize=(9, 6))

    categories = ['정상', '노킹']
    counts = [NORMAL_SAMPLES, KNOCKING_SAMPLES]
    colors = ['#4CAF50', '#F44336']

    bars = ax.bar(categories, counts, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=2.5, width=0.5)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = count / TOTAL_SAMPLES * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{count}개\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=15, fontweight='bold')

    ax.set_ylabel('샘플 수', fontsize=14, fontweight='bold')
    ax.set_title(f'학습 데이터셋 구성 (총 {TOTAL_SAMPLES}개)', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 600)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 데이터 출처
    ax.text(0.5, -0.15, 'AudioSet + YouTube + 직접 녹음 + 데이터 증강',
            ha='center', transform=ax.transAxes, fontsize=11,
            style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / '3_dataset.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 3_dataset.png")
    plt.close()


def generate_training_process():
    """학습 과정 설명"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # 3단계 프로세스
    process = [
        ('1단계\nYAMNet 로드', 0.8, '#E3F2FD',
         'TensorFlow Hub에서\nYAMNet 모델 로드\n(Freeze)'),
        ('2단계\n임베딩 추출', 0.55, '#90CAF9',
         '698개 오디오 파일에서\n1024차원 임베딩 추출\n(평균 pooling)'),
        ('3단계\n커스텀 학습', 0.3, '#66BB6A',
         'Dense Layer 학습\n(Adam optimizer)\n(Early Stopping)'),
    ]

    for i, (title, y_pos, color, desc) in enumerate(process):
        x_pos = 0.15 + (i * 0.35)

        # 큰 박스
        rect = mpatches.FancyBboxPatch((x_pos - 0.12, y_pos - 0.15), 0.24, 0.3,
                                       boxstyle="round,pad=0.02",
                                       edgecolor='black', facecolor=color,
                                       linewidth=3, alpha=0.9,
                                       transform=ax.transAxes)
        ax.add_patch(rect)

        # 제목
        ax.text(x_pos, y_pos + 0.1, title, ha='center', va='center',
                fontsize=14, fontweight='bold', transform=ax.transAxes)

        # 설명
        ax.text(x_pos, y_pos - 0.05, desc, ha='center', va='center',
                fontsize=10, transform=ax.transAxes, linespacing=1.5)

        # 화살표
        if i < len(process) - 1:
            ax.annotate('', xy=(x_pos + 0.2, y_pos),
                       xytext=(x_pos + 0.13, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'),
                       transform=ax.transAxes)

    # 제목
    ax.text(0.5, 0.95, 'Transfer Learning 학습 과정', ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / '4_training_process.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 4_training_process.png")
    plt.close()


def generate_comparison():
    """기존 vs 새 시스템 비교"""
    fig, ax = plt.subplots(figsize=(12, 7))

    systems = ['기존\n(Random Forest)', '새 시스템\n(Transfer Learning)']
    metrics = {
        'YAMNet\n활용도': [5, 100],
        '딥러닝\n여부': [0, 100],
        '설명\n용이성': [70, 90]
    }

    x = np.arange(len(systems))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * (i - 1)
        bars = ax.bar(x + offset, values, width, label=metric,
                     color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)

        # 값 표시
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{val}%', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

    ax.set_ylabel('상대적 점수 (%)', fontsize=13, fontweight='bold')
    ax.set_title('기존 시스템 vs Transfer Learning 비교', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / '5_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 5_comparison.png")
    plt.close()


def main():
    print("="*70)
    print("📊 YAMNet Transfer Learning 포스터 그래프 생성")
    print("="*70)

    generate_system_architecture()
    generate_model_structure()
    generate_dataset_distribution()
    generate_training_process()
    generate_comparison()

    print("\n" + "="*70)
    print("✅ 모든 그래프 생성 완료!")
    print("="*70)
    print(f"📁 저장 위치: {output_dir.absolute()}\n")
    print("생성된 그래프:")
    print("  1. 1_system_architecture.png - 시스템 아키텍처")
    print("  2. 2_model_structure.png - 모델 구조도")
    print("  3. 3_dataset.png - 데이터셋 분포")
    print("  4. 4_training_process.png - 학습 과정")
    print("  5. 5_comparison.png - 기존 vs 새 시스템 비교")
    print("\n💡 포스터 추천:")
    print("  - 필수: 1번(아키텍처), 2번(모델 구조), 3번(데이터셋)")
    print("  - 선택: 4번(학습 과정) 또는 5번(비교)")


if __name__ == "__main__":
    main()
