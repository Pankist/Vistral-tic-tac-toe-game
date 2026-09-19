"""Announcer phrases for submarine game."""

PHRASES = {
    "start": ["Starting puzzle monitor.", "Watching for puzzles.", "Ready to recognize puzzles."],
    "monitoring": ["Monitoring for puzzles..."],
    "settled": ["Image settled, analyzing..."],
    "no_puzzle": ["No puzzle detected. Waiting for changes."],
    "recognizing": ["Recognized a puzzle. Reading question and options..."],
    "solving": ["Recognized a puzzle. Provided {option_count} options to choose from. Picking the right one..."],
    "solved": ["Answer: {answer}"],
    "error": ["Could not process puzzle."],
    "standby": ["Waiting for next puzzle."],
}
