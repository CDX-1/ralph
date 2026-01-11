#!/usr/bin/env python3
"""
Test script to verify object detection and decision logic.
Prints detailed debug info about what the robot sees and why it makes decisions.
"""

import sys
sys.path.insert(0, 'src')

# Test various scenarios
test_scenarios = [
    {
        "name": "Large object close in middle",
        "M_objects": [(0.6, 0.20)],  # 0.6m away, 20% of frame
        "expected": "STOP or STEER"
    },
    {
        "name": "Small object far in middle", 
        "M_objects": [(3.0, 0.02)],  # 3m away, 2% of frame
        "expected": "FORWARD (should ignore)"
    },
    {
        "name": "Medium object medium distance in middle",
        "M_objects": [(1.2, 0.10)],  # 1.2m away, 10% of frame
        "expected": "STEER (should avoid)"
    },
    {
        "name": "Large object on left, clear right",
        "L_objects": [(0.7, 0.15)],  # Large object close on left
        "expected": "STEER_RIGHT"
    },
    {
        "name": "Multiple small objects in middle",
        "M_objects": [(1.5, 0.04), (1.8, 0.03), (2.0, 0.05)],
        "expected": "STEER (cumulative threat)"
    },
    {
        "name": "Very large object medium distance",
        "M_objects": [(1.5, 0.25)],  # 1.5m but huge (25% of frame)
        "expected": "STEER (size matters)"
    },
]

def run_test(scenario, decide_action):
    print(f"\n{'='*60}")
    print(f"TEST: {scenario['name']}")
    print(f"Expected: {scenario['expected']}")
    print('='*60)
    
    LL = scenario.get('LL_objects', [])
    L = scenario.get('L_objects', [])
    M = scenario.get('M_objects', [])
    R = scenario.get('R_objects', [])
    RR = scenario.get('RR_objects', [])
    
    action, _, meta = decide_action(LL, L, M, R, RR, "RIGHT", None, verbose=True)
    
    print(f"\n>>> RESULT: {action}")
    return action

if __name__ == "__main__":
    try:
        from vision import decide_action
        print("Testing improved detection logic with distance + bounding box size...\n")
        
        for scenario in test_scenarios:
            action = run_test(scenario, decide_action)
        
        print("\n" + "="*60)
        print("All tests complete!")
        print("="*60)
        
    except ImportError as e:
        print(f"Error: Could not import vision module: {e}")
        print("Make sure you've replaced src/vision.py with the improved version")
        sys.exit(1)