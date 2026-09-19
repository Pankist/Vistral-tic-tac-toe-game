"""LLM prompts for submarine puzzle recognition and solving."""


def recognizer_system() -> str:
    return """You are a professional hiring manager assessing assignments to understand if a hiring process is correct and good.

You will be presented different puzzles:
- Psychotest American style ones
- Missed words, word selection from suggestions
- Mathematical puzzles
- Logical puzzles of different descriptions
- Small coding assignments (what does this code print, what message is displayed)
- Small engineering questions (what function does something, which AWS service fits best)

Analyze and solve each puzzle professionally.

Return ONLY this JSON:
{
  "status": "recognized" | "solved" | "not_recognized",
  "puzzle_description": "textual description of what the puzzle asks for",
  "answer": "answer to the puzzle so the other person understands you answered correctly"
}

Be professional and accurate."""


def recognizer_user(image_data: str) -> list[dict]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_data}
        },
        {
            "type": "text",
            "text": "Analyze this assignment and solve it. Return JSON only."
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
