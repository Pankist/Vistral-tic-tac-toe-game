"""LLM prompts for submarine puzzle recognition and solving."""


def recognizer_system() -> str:
    return """Look at the image and solve the puzzle.

Return ONLY this JSON:
{
  "has_puzzle": true/false,
  "question": "what the assignment is asking you to do",
  "answer": "the correct answer"
}

Simple. Analyze the visual patterns and give the answer."""


def recognizer_user(image_data: str) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_data}
        },
        {
            "type": "text",
            "text": "What is the question in this image, and what is the correct answer? Return JSON only."
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
