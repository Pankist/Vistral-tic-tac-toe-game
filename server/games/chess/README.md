# chess/

The next game lives here: a `game.py` implementing `core.protocols.Game`
(state, legality, terminality, serialization) plus a `marks.py` for reading
pieces from the rectified board. Empty on purpose — the seam is the point.
The generic LLM engine (`core/engine_llm.py`) would already play it.
