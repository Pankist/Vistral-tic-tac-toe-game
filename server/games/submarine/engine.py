"""Submarine puzzle solver engine using Anthropic Vision API."""

import base64
import cv2
import numpy as np

from server.core import config as cfg
from server.core.llm import AnthropicClient, parse_strict_json, LLMUnavailable
from server.games.submarine import prompts
from server.games.submarine.config import MODEL_RECOGNIZER, MODEL_SOLVER, VISION_TIMEOUT_S


class SubmarineEngine:
    """Vision-based puzzle recognizer and solver."""

    name = "submarine_solver"

    def __init__(self, client: AnthropicClient = None, log=None):
        self.client = client or AnthropicClient()
        self.log = log or (lambda *_: None)

    def recognize_puzzle(self, image: np.ndarray) -> dict:
        """Use vision to recognize puzzle from image.

        Returns: {has_puzzle: bool, puzzle_type: str, puzzle_text: str, options: list}
        """
        # Convert image to base64 data URL
        _, buffer = cv2.imencode('.jpg', image)
        b64 = base64.b64encode(buffer).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{b64}"

        try:
            messages = [
                {"role": "system", "content": prompts.recognizer_system()},
                {"role": "user", "content": prompts.recognizer_user(image_url)}
            ]

            result = self.client.chat(MODEL_RECOGNIZER, messages, timeout=VISION_TIMEOUT_S)
            self.log("llm_call", {
                "model": MODEL_RECOGNIZER,
                "purpose": "recognize_puzzle",
                "latency_ms": result["latency_ms"],
                "cost": result.get("cost", {}),
            })

            recognition = parse_strict_json(result["content"])
            return recognition

        except (LLMUnavailable, ValueError) as e:
            error_msg = str(e)
            self.log("error", {
                "type": "recognition_failed",
                "error": error_msg,
                "model": MODEL_RECOGNIZER
            })
            print(f"[SUBMARINE ERROR] Recognition failed: {error_msg}")  # Console logging
            return {
                "status": "not_recognized",
                "puzzle_description": f"API Error: {error_msg[:100]}",
                "answer": "Unable to process - check API credits"
            }

    def solve_puzzle(self, puzzle_text: str, options: list[str] = None) -> dict:
        """Solve the recognized puzzle.

        Returns: {answer: str, reasoning: str, confidence: float}
        """
        try:
            messages = [
                {"role": "system", "content": prompts.solver_system()},
                {"role": "user", "content": prompts.solver_user(puzzle_text, options)}
            ]

            result = self.client.chat(MODEL_SOLVER, messages, timeout=VISION_TIMEOUT_S)
            self.log("llm_call", {
                "model": MODEL_SOLVER,
                "purpose": "solve_puzzle",
                "latency_ms": result["latency_ms"],
                "cost": result.get("cost", {}),
            })

            solution = parse_strict_json(result["content"])
            return solution

        except (LLMUnavailable, ValueError) as e:
            self.log("error", {"type": "solving_failed", "error": str(e)})
            return {
                "answer": "Error solving puzzle",
                "reasoning": str(e),
                "confidence": 0.0
            }

    def decide(self, board: dict, history: list) -> int:
        """Compatibility method for Engine protocol - triggers solve."""
        return 0
