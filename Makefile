PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

setup:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

dev:
	$(BIN)/uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

run:
	$(BIN)/uvicorn server.main:app --host 0.0.0.0 --port 8000

test:
	$(BIN)/pytest tests/ -q

print-board:
	$(BIN)/python scripts/make_board.py

client:
	cd client && $(PY) -m http.server 3000

.PHONY: setup dev run test print-board client
