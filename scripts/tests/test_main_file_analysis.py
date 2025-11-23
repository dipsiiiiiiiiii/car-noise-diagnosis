#!/usr/bin/env python3
"""
Test main.py's file analysis directly
"""

import sys
from pathlib import Path

# Import main module
sys.path.insert(0, str(Path(__file__).parent))
from main import CarNoiseDiagnosisSystem

# Test
system = CarNoiseDiagnosisSystem(debug_mode=False, comparison_mode=False)
result = system.analyze_audio_file("data/testing/test_verified.wav")

print("\n=== TEST RESULT ===")
print(f"Success: {result.get('success')}")
if result.get('success'):
    diagnosis = result['diagnosis']
    print(f"Issues: {len(diagnosis.get('issues', []))}")
    print(f"Confidence: {diagnosis.get('confidence', 0):.1%}")
    if diagnosis.get('issues'):
        print(f"Description: {diagnosis['issues'][0].get('description')}")
    else:
        print("No knocking detected")
