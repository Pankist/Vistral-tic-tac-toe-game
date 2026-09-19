"""LLM prompts for submarine puzzle recognition and solving."""


def recognizer_system() -> str:
    return """You are a puzzle solver. Analyze the image for any logical puzzles, IQ test questions, or challenges.
If you find a puzzle, SOLVE IT by analyzing the visual patterns in the image directly.

Output ONLY valid JSON in this exact format:
{
  "has_puzzle": true/false,
  "puzzle_type": "logical" | "math" | "pattern" | "multiple_choice" | "riddle" | "none",
  "puzzle_text": "exact text of the question/puzzle (for display only)",
  "options": ["A", "B", "C", "D"] (if multiple choice, else empty array),
  "answer": "your answer (the correct option letter or answer)",
  "reasoning": "brief explanation of your solution process"
}

If no clear puzzle is visible, set has_puzzle to false and leave answer/reasoning empty.
IMPORTANT: Solve the puzzle by analyzing the IMAGE directly - look at visual patterns, sequences, relationships.
The puzzle_text is just for display - your answer should come from visual analysis."""


def recognizer_user(image_data: str) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_data}
        },
        {
            "type": "text",
            "text": "Analyze this image. If there's a puzzle, solve it by examining the visual patterns. Return JSON only."
        }
    ]


def solver_system() -> str:
    return """You are a logical puzzle solver. Given a puzzle, provide a concise, correct answer.

Output ONLY valid JSON in this exact format:
{
  "answer": "your answer here",
  "reasoning": "brief explanation of your solution process",
  "confidence": 0.0-1.0
}

Be accurate and concise. For multiple choice, give the letter and brief explanation."""


def solver_user(puzzle_text: str, options: list[str] = None) -> str:
    opts = f"\nOptions: {', '.join(options)}" if options else ""
    return f"Solve this puzzle:\n{puzzle_text}{opts}\n\nReturn JSON only."
