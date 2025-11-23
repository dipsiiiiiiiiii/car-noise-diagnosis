#!/usr/bin/env python3
"""
모델 전환 기능 테스트
"""
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from diagnosis.analyzer import CarNoiseDiagnoser

def test_model_switch():
    """Test model switching functionality"""
    print("=" * 60)
    print("모델 전환 기능 테스트")
    print("=" * 60)

    # Find custom model path
    verified_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_verified.pkl"
    oneclass_v4_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v4.pkl"
    oneclass_v2_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass_v2.pkl"
    oneclass_model_path = Path(__file__).parent / "data" / "models" / "car_classifier_oneclass.pkl"

    if verified_path.exists():
        model_path = str(verified_path)
    elif oneclass_v4_path.exists():
        model_path = str(oneclass_v4_path)
    elif oneclass_v2_path.exists():
        model_path = str(oneclass_v2_path)
    elif oneclass_model_path.exists():
        model_path = str(oneclass_model_path)
    else:
        model_path = None
        print("❌ 테스트 실패: 커스텀 모델을 찾을 수 없습니다.")
        return False

    print(f"\n📁 사용할 모델: {Path(model_path).name}")
    print("-" * 60)

    # Test 1: Initialize with Custom model
    print("\n[테스트 1] Custom 모델로 초기화")
    diagnoser = CarNoiseDiagnoser(model_path=model_path, use_custom_model=True)
    assert diagnoser.mode == "custom", f"❌ 실패: mode={diagnoser.mode}, 예상=custom"
    assert diagnoser.custom_classifier is not None, "❌ 실패: custom_classifier가 None입니다"
    print(f"✅ 성공: mode={diagnoser.mode}")

    # Test 2: Switch to Baseline
    print("\n[테스트 2] Baseline으로 전환")
    mode = diagnoser.switch_model(use_custom=False)
    assert mode == "baseline", f"❌ 실패: mode={mode}, 예상=baseline"
    assert diagnoser.mode == "baseline", f"❌ 실패: diagnoser.mode={diagnoser.mode}"
    print(f"✅ 성공: mode={diagnoser.mode}")

    # Test 3: Switch back to Custom
    print("\n[테스트 3] Custom으로 다시 전환")
    mode = diagnoser.switch_model(use_custom=True)
    assert mode == "custom", f"❌ 실패: mode={mode}, 예상=custom"
    assert diagnoser.mode == "custom", f"❌ 실패: diagnoser.mode={diagnoser.mode}"
    print(f"✅ 성공: mode={diagnoser.mode}")

    # Test 4: Initialize with Baseline
    print("\n[테스트 4] Baseline으로 초기화")
    diagnoser2 = CarNoiseDiagnoser(model_path=model_path, use_custom_model=False)
    assert diagnoser2.mode == "baseline", f"❌ 실패: mode={diagnoser2.mode}, 예상=baseline"
    print(f"✅ 성공: mode={diagnoser2.mode}")

    # Test 5: Initialize without model path
    print("\n[테스트 5] 모델 경로 없이 초기화 (Baseline만 사용)")
    diagnoser3 = CarNoiseDiagnoser(model_path=None)
    assert diagnoser3.mode == "baseline", f"❌ 실패: mode={diagnoser3.mode}, 예상=baseline"
    assert diagnoser3.custom_classifier is None, "❌ 실패: custom_classifier가 None이 아닙니다"
    print(f"✅ 성공: mode={diagnoser3.mode}")

    print("\n" + "=" * 60)
    print("✅ 모든 테스트 통과!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = test_model_switch()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
