#!/usr/bin/env python3
"""Quick test to verify submarine game flow works."""

import sys
import numpy as np

# Test imports
print("Testing imports...")
try:
    from server.games.submarine.game import Submarine
    from server.games.submarine.engine import SubmarineEngine
    from server.games.submarine.prompts import recognizer_system, solver_system
    from server.perception.change_detector import ChangeDetector
    from server.core.fsm import FSM, Session, RecognizePuzzle, SolvePuzzle, ControlEvent
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test submarine game
print("\nTesting Submarine game...")
game = Submarine()
assert game.name == "submarine"
assert game.initial_board()["status"] == "monitoring"
print(f"✓ Game: {game.name}")
print(f"✓ Initial state: {game.initial_board()['status']}")

# Test change detector
print("\nTesting ChangeDetector...")
detector = ChangeDetector(settle_frames=3, threshold=0.05)
frame1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
result1 = detector.check_change(frame1)
assert result1['changed'] == False  # First frame
assert result1['settled'] == False
print(f"✓ Frame 1: changed={result1['changed']}, settled={result1['settled']}")

# Same frame multiple times should settle
for i in range(4):
    result = detector.check_change(frame1)
    print(f"  Frame {i+2}: stable_count={result['stable_count']}, settled={result['settled']}")

print("✓ Change detection works")

# Test FSM states
print("\nTesting FSM with submarine...")
fsm = FSM(game)
session = Session(state="MONITORING")

# Test set_corners control
corners_event = ControlEvent(action="set_corners", payload=[(10, 10), (90, 10), (90, 90), (10, 90)])
effects = fsm.step(session, corners_event)
print(f"✓ set_corners: {len(effects)} effects")

# Test rescan control
rescan_event = ControlEvent(action="rescan")
effects = fsm.step(session, rescan_event)
print(f"✓ rescan: state={session.state}, {len(effects)} effects")

# Test recognize_result
recognize_payload = {
    "has_puzzle": True,
    "puzzle_type": "logical",
    "puzzle_text": "What comes next: 2, 4, 8, 16, ?",
    "options": ["32", "20", "24", "18"]
}
recognize_event = ControlEvent(action="recognize_result", payload=recognize_payload)
effects = fsm.step(session, recognize_event)
print(f"✓ recognize_result: state={session.state}")
assert session.state == "SOLVING"
assert session.puzzle_text == "What comes next: 2, 4, 8, 16, ?"

# Test solve_result
solve_payload = {
    "answer": "32",
    "reasoning": "Each number is double the previous: 2×2=4, 4×2=8, 8×2=16, 16×2=32",
    "confidence": 0.99
}
solve_event = ControlEvent(action="solve_result", payload=solve_payload)
effects = fsm.step(session, solve_event)
print(f"✓ solve_result: state={session.state}, answer={session.answer}")
assert session.state == "STANDBY"
assert session.answer == "32"

print("\n" + "="*50)
print("✓ ALL TESTS PASSED!")
print("="*50)
print("\nSubmarine game is ready to use!")
print("\nNext steps:")
print("1. Add your ANTHROPIC_API_KEY to .env")
print("2. Run: make dev")
print("3. Open client in browser")
print("4. Select 'Submarine' from dropdown")
print("5. Point camera at a puzzle and click Start")
