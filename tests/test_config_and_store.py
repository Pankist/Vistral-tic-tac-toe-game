import json

from server.core import config as cfg
from server.core.store import CLAMPS, Store


def test_config_namespaces_are_disjoint():
    merged = cfg.merged()          # raises AssertionError on collision
    assert "K_STABLE" in merged and "T_EMPTY" in merged


def test_config_hash_stable_and_secret_free():
    h1, h2 = cfg.config_hash(), cfg.config_hash()
    assert h1 == h2 and len(h1) == 12


def test_store_events_and_provenance(tmp_path):
    store = Store("tic_tac_toe", root=tmp_path)
    store.log_decision("engine_move", {"move": 4})
    lines = (tmp_path / "tic_tac_toe" / "events.jsonl").read_text().splitlines()
    rec = json.loads(lines[-1])
    assert rec["move"] == 4
    assert set(rec["provenance"]) == {"git_sha", "config_hash", "memory_version"}


def test_memory_survives_reload(tmp_path):
    store = Store("tic_tac_toe", root=tmp_path)
    store.finish_game("draw", 9, misreads=1, arbiter_calls=2, mismatches=0)
    again = Store("tic_tac_toe", root=tmp_path)
    assert again.memory["games_played"] == 1
    assert again.memory["arbiter_calls"] == 2


def test_corrupt_memory_falls_back_to_defaults(tmp_path):
    d = tmp_path / "tic_tac_toe"
    d.mkdir(parents=True)
    (d / "memory.json").write_text("{not json")
    store = Store("tic_tac_toe", root=tmp_path)
    assert store.memory["games_played"] == 0


def test_learned_t_empty_is_clamped(tmp_path):
    store = Store("tic_tac_toe", root=tmp_path)
    store.memory["t_empty"] = 0.5              # drifted / corrupted value
    lo, hi = CLAMPS["t_empty"]
    assert store.learned_t_empty() == hi
    store.memory["t_empty"] = -3
    assert store.learned_t_empty() == lo


def test_t_empty_learning_from_samples(tmp_path):
    store = Store("tic_tac_toe", root=tmp_path)
    store.observe_empty_ink([0.012] * 40)
    store.finish_game("draw", 9, 0, 0, 0)
    v = store.memory["t_empty"]
    lo, hi = CLAMPS["t_empty"]
    assert v is not None and lo <= v <= hi
