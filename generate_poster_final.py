#!/usr/bin/env python3
"""
포스터 발표용 핵심 그래프 생성 (실제 데이터 기반)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
output_dir = Path("poster_final")
output_dir.mkdir(exist_ok=True)

# === 실제 데이터 ===
TOTAL_SAMPLES = 698
NORMAL_SAMPLES = 182
KNOCKING_SAMPLES = 516
TEST_ACCURACY = 100.0
TEST_SAMPLES = 140
TEST_NORMAL = 37
TEST_KNOCKING = 103
KNOCKING_AVG_PROB = 97.7
NORMAL_AVG_PROB = 3.6
THRESHOLD = 75


def generate_system_architecture():
    """시스템 아키텍처 (핵심만)"""
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.axis('off')

    arrow_props = dict(arrowstyle='->', lw=3, color='black')

    # 단계별 박스
    stages = [
        (0.5, 0.92, '마이크 입력', '#E3F2FD'),
        (0.5, 0.82, 'YAMNet (521 클래스)', '#90CAF9'),
        (0.5, 0.72, '88차원 임베딩 추출', '#64B5F6'),
        (0.5, 0.62, 'Random Forest (100 trees)', '#42A5F5'),
        (0.5, 0.52, '확률 75% 기준 판정', '#FFA726'),
        (0.5, 0.42, '진단 결과', '#4CAF50'),
    ]

    for i, (x, y, text, color) in enumerate(stages):
        bbox = dict(boxstyle='round,pad=0.7', facecolor=color,
                   edgecolor='black', linewidth=3, alpha=0.9)
        ax.text(x, y, text, ha='center', va='center', fontsize=14,
                fontweight='bold', bbox=bbox, transform=ax.transAxes)

        if i < len(stages) - 1:
            ax.annotate('', xy=(x, stages[i+1][1] + 0.04),
                       xytext=(x, y - 0.04),
                       arrowprops=arrow_props, transform=ax.transAxes)

    ax.text(0.5, 0.97, 'YAMNet 기반 하이브리드 진단 시스템', ha='center', va='top',
            fontsize=16, fontweight='bold', transform=ax.transAxes)

    # 핵심 지표
    metrics_text = (
        "📊 핵심 성능 지표\n\n"
        f"• 학습 데이터: {TOTAL_SAMPLES}개 (정상 {NORMAL_SAMPLES}, 노킹 {KNOCKING_SAMPLES})\n"
        f"• 테스트 정확도: {TEST_ACCURACY}% (n={TEST_SAMPLES})\n"
        f"• 노킹 판정 기준: 확률 {THRESHOLD}% 이상\n"
        f"• 노킹 샘플 평균: {KNOCKING_AVG_PROB}%\n"
        f"• 정상 샘플 평균: {NORMAL_AVG_PROB}%\n"
        "• 모델 크기: 216KB (경량화)"
    )
    bbox_metrics = dict(boxstyle='round,pad=1', facecolor='#FFF9C4',
                       edgecolor='black', linewidth=2, alpha=0.95)
    ax.text(0.5, 0.18, metrics_text, ha='center', va='center',
            fontsize=11, bbox=bbox_metrics, transform=ax.transAxes,
            linespacing=1.8)

    plt.tight_layout()
    plt.savefig(output_dir / '1_system_overview.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 1_system_overview.png (시스템 구조 + 핵심 지표)")
    plt.close()


def generate_confusion_matrix():
    """혼동 행렬 (실제 데이터)"""
    fig, ax = plt.subplots(figsize=(9, 8))

    # 실제 데이터
    cm = np.array([
        [TEST_NORMAL, 0],
        [0, TEST_KNOCKING]
    ])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': '샘플 수'},
                xticklabels=['정상 예측', '노킹 예측'],
                yticklabels=['정상 실제', '노킹 실제'],
                linewidths=3, linecolor='black',
                annot_kws={'fontsize': 22, 'fontweight': 'bold'},
                ax=ax, vmin=0, vmax=TEST_KNOCKING)

    ax.set_title(f'테스트 결과 (n={TEST_SAMPLES})\n정확도: {TEST_ACCURACY}%',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('예측', fontsize=15, fontweight='bold')
    ax.set_ylabel('실제', fontsize=15, fontweight='bold')

    # 추가 정보
    info_text = f'오분류: 0개 (FP=0, FN=0)'
    ax.text(0.5, -0.15, info_text, ha='center', transform=ax.transAxes,
            fontsize=13, fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(output_dir / '2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 2_confusion_matrix.png (혼동 행렬)")
    plt.close()


def generate_probability_distribution():
    """확률 분포 비교"""
    fig, ax = plt.subplots(figsize=(10, 7))

    categories = ['정상 샘플\n(평균 확률)', '판정 기준\n(Threshold)', '노킹 샘플\n(평균 확률)']
    values = [NORMAL_AVG_PROB, THRESHOLD, KNOCKING_AVG_PROB]
    colors = ['#4CAF50', '#FFA726', '#F44336']

    bars = ax.bar(categories, values, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=2.5, width=0.6)

    # 값 표시
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val}%', ha='center', va='bottom',
                fontsize=18, fontweight='bold')

    # 판정 기준선
    ax.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2.5, alpha=0.6)
    ax.text(2.5, THRESHOLD + 3, f'{THRESHOLD}% 기준',
            fontsize=12, fontweight='bold', color='red')

    ax.set_ylabel('노킹 확률 (%)', fontsize=14, fontweight='bold')
    ax.set_title('노킹 판정 확률 분포', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 설명 추가
    ax.text(0.5, -0.12,
            '정상 샘플은 기준(75%) 훨씬 아래, 노킹 샘플은 훨씬 위 → 명확한 구분',
            ha='center', transform=ax.transAxes,
            fontsize=11, style='italic', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / '3_probability_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 3_probability_distribution.png (확률 분포)")
    plt.close()


def generate_dataset_info():
    """데이터셋 구성"""
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

    plt.tight_layout()
    plt.savefig(output_dir / '4_dataset.png', dpi=300, bbox_inches='tight')
    print(f"✅ 생성: 4_dataset.png (데이터셋 구성)")
    plt.close()


def main():
    print("="*70)
    print("📊 포스터 최종 그래프 생성 (실제 데이터 기반)")
    print("="*70)
    print("\n실제 데이터:")
    print(f"  • 총 학습 샘플: {TOTAL_SAMPLES}개 (정상 {NORMAL_SAMPLES}, 노킹 {KNOCKING_SAMPLES})")
    print(f"  • 테스트 정확도: {TEST_ACCURACY}% (n={TEST_SAMPLES})")
    print(f"  • 노킹 판정 기준: {THRESHOLD}% 이상")
    print(f"  • 평균 확률: 정상 {NORMAL_AVG_PROB}%, 노킹 {KNOCKING_AVG_PROB}%")
    print()

    generate_system_architecture()
    generate_confusion_matrix()
    generate_probability_distribution()
    generate_dataset_info()

    print("\n" + "="*70)
    print("✅ 최종 그래프 생성 완료!")
    print("="*70)
    print(f"📁 저장 위치: {output_dir.absolute()}\n")
    print("생성된 그래프 (포스터 추천 순서):")
    print("  1. 1_system_overview.png     - 시스템 구조 + 핵심 지표 (필수)")
    print("  2. 2_confusion_matrix.png    - 혼동 행렬 (필수)")
    print("  3. 3_probability_distribution.png - 확률 분포 (필수)")
    print("  4. 4_dataset.png             - 데이터셋 구성 (선택)\n")
    print("💡 포스터 레이아웃 제안:")
    print("  [상단] 제목 + 배경")
    print("  [좌측] 1번 그래프 (시스템 + 지표)")
    print("  [중앙] 2번 그래프 (혼동 행렬)")
    print("  [우측] 3번 그래프 (확률 분포)")
    print("  [하단] 결론 및 향후 계획")


if __name__ == "__main__":
    main()
