"""Full simulated game over the real WebSocket — end-to-end smoke test.

Connects to a running backend (local or ALB), drives the dev simulate-mark
path through a complete game: calibration, X moves, AWAIT_DRAW confirmations
of the agent's O, through GAME_OVER. Perfect-play engine on both scripts
means the expected result is a draw.

Run: .venv/bin/python scripts/ws_smoke.py [ws://host/ws]
"""

import asyncio
import json
import sys

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws"


async def main() -> None:
    spoken, states = [], []
    async with websockets.connect(URL, open_timeout=15) as ws:

        async def drain(until_states):
            """Read messages until the FSM lands in one of `until_states`."""
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                if msg["type"] == "speak":
                    spoken.append(msg["text"])
                    print(f"  agent: {msg['text']}")
                elif msg["type"] == "state":
                    states.append(msg)
                    if msg["fsm"] in until_states:
                        return msg

        await drain({"IDLE"})
        print(f"connected to {URL}")

        # human X keeps choosing the first free cell; agent replies via minimax
        for _ in range(5):
            state = states[-1]
            free = [i for i, m in enumerate(state["board"]) if not m]
            await ws.send(json.dumps({"type": "control",
                                      "action": "simulate_mark",
                                      "cell": free[0]}))
            state = await drain({"AWAIT_DRAW", "GAME_OVER"})
            if state["fsm"] == "GAME_OVER":
                break
            await ws.send(json.dumps({"type": "control",
                                      "action": "simulate_mark",
                                      "cell": state["expected_cell"]}))
            state = await drain({"HUMAN_TURN", "GAME_OVER"})
            if state["fsm"] == "GAME_OVER":
                break

    final = states[-1]
    assert final["fsm"] == "GAME_OVER", f"game did not finish: {final['fsm']}"
    assert len(spoken) >= 5, "expected narration at every announce point"
    board = final["board"]
    print(f"final board: {board}")
    print(f"utterances: {len(spoken)}, states seen: {len(states)}")
    print("PASS — full game over WebSocket, GAME_OVER reached")


if __name__ == "__main__":
    asyncio.run(main())
