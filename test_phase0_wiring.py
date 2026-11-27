"""Phase 0 wiring regressions for the FastAPI membrane.

Covers the broken connections fixed in Phase 0:
  B2/B3  /observe drives the articulatory voice and schedules a WS broadcast
  B4     model-mutating paths are serialised with the runtime loop
  B5     /observe(train) and /train-step advance the recurrent state once
  B6     memory salience reaches the Z³ input vector
  B7     the optimizer survives learning-rate changes (moments + param groups)

Run with:
    python test_phase0_wiring.py
"""
from __future__ import annotations

import os
import tempfile
import threading

os.environ.setdefault("Z3_STATE_DIR", tempfile.mkdtemp(prefix="z3_phase0_"))
os.environ.setdefault("DISABLE_REMOTE_CORPUS_STREAM", "1")
os.environ.setdefault("LANGUAGE_TRAINING_MIN_WORDS", "1")
os.environ.setdefault("Z3_CORPUS_MIN_WORDS", "1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: object = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} -- {detail}")


print("=" * 60)
print("TESTING PHASE 0 WIRING")
print("=" * 60)

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402

# --- B6: salience reaches Z³ -------------------------------------------------
world = {
    "latent_state": [0.1 * i for i in range(8)],
    "total_loss": 9.99,
    "prediction_loss": 0.2,
    "reconstruction_loss": 0.3,
    "memory_loss": 0.4,
    "coherence_alignment_loss": 0.5,
    "novelty": 0.6,
    "nearest_memory_distance": 0.7,
}
memory = {"reconstruction_confidence": 0.8, "salience": 0.42}
vec = main._compose_z3_input(world, memory, 16)
check("Composed input has input_dim length", len(vec) == 16, len(vec))
check("Memory salience is present in the Z³ input", 0.42 in vec, vec)
check("Redundant world total_loss no longer occupies a slot", 9.99 not in vec, vec)
check("All latent dims and scalar signals are kept at default dims", vec[:8] == world["latent_state"] and vec[9:] == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec)
wide = dict(world, latent_state=[1.0] * 12)
wide_vec = main._compose_z3_input(wide, memory, 16)
check("Wide latent is trimmed, not the scalar signals", len(wide_vec) == 16 and wide_vec[-8:] == [0.42, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], wide_vec)

with TestClient(main.app) as client:
    model = main.get_model()

    # --- B7: optimizer persistence ------------------------------------------
    opt_a = main._get_optimizer(model, 1e-3)
    sentinel = torch.nn.Parameter(torch.zeros(1))
    opt_a.add_param_group({"params": [sentinel], "lr": 1e-3})
    opt_b = main._get_optimizer(model, 5e-4)
    check("Optimizer object survives a learning-rate change", opt_a is opt_b)
    check("Learning-rate change is applied to every param group", all(abs(g["lr"] - 5e-4) < 1e-12 for g in opt_b.param_groups), [g["lr"] for g in opt_b.param_groups])
    check("Extra param groups (decoder) are preserved", any(sentinel is p for g in opt_b.param_groups for p in g["params"]))
    # remove the sentinel group again (only if it survived)
    opt_b.param_groups[:] = [g for g in opt_b.param_groups if not any(p is sentinel for p in g["params"])]

    # --- B5: single recurrent advance per trained observation ---------------
    commits = {"n": 0}
    original_commit = model._commit_state

    def counting_commit(*args, **kwargs):
        commits["n"] += 1
        return original_commit(*args, **kwargs)

    model._commit_state = counting_commit  # type: ignore[assignment]
    try:
        r = client.post("/observe", json={"observation": {"text": "phase zero", "value": 1.0}, "domain": "test", "train": True, "persist": False})
        check("/observe(train) succeeds", r.status_code == 200, r.text[:300])
        observe_commits = commits["n"]
        commits["n"] = 0
        r2 = client.post("/train-step", json={"x": [0.1] * model.config.input_dim, "persist": False})
        check("/train-step succeeds", r2.status_code == 200, r2.text[:300])
        train_commits = commits["n"]
    finally:
        model._commit_state = original_commit  # type: ignore[assignment]
    check("/observe(train) commits recurrent state exactly once", observe_commits == 1, observe_commits)
    check("/train-step commits recurrent state exactly once", train_commits == 1, train_commits)

    # --- B2/B3: voice push from /observe ------------------------------------
    sent = []
    manager = main.get_ws_manager()
    original_send = manager.send_json

    async def capture_send(payload):
        sent.append(payload)

    manager.send_json = capture_send  # type: ignore[assignment]
    try:
        r = client.post("/observe", json={"observation": {"text": "voice check", "value": 2.0}, "domain": "test", "train": False, "persist": False})
        voice = r.json().get("z3", {}).get("voice", {})
        if main.get_articulatory_voice is not None:
            check("/observe drives the articulatory voice", voice.get("articulatory") is True, voice)
            check("/observe schedules the WebSocket broadcast on the main loop", voice.get("broadcast") is True, voice)
            # give the event loop a moment to run the scheduled coroutine
            client.get("/health")
            check("Broadcast payload reaches the WebSocket manager", len(sent) >= 1, len(sent))
        else:
            check("Synth push still reported without articulatory module", "synth" in voice, voice)
    finally:
        manager.send_json = original_send  # type: ignore[assignment]

    # --- B4: concurrent heartbeat + API mutation ----------------------------
    errors = []
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            try:
                main._runtime_tick()
            except Exception as exc:  # pragma: no cover - surfaced via check
                errors.append(repr(exc))
                return

    check("Runtime lock is re-entrant", isinstance(main._RUNTIME_LOCK, type(threading.RLock())))
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    statuses = []
    for i in range(12):
        statuses.append(client.post("/observe", json={"observation": {"text": f"race {i}", "value": float(i)}, "domain": "test", "train": i % 2 == 0, "persist": False}).status_code)
        statuses.append(client.post("/train-step", json={"x": [0.05 * i] * model.config.input_dim, "persist": False}).status_code)
        statuses.append(client.post("/step", json={"x": [0.01 * i] * model.config.input_dim, "update_state": True, "persist": False}).status_code)
    stop.set()
    worker.join(timeout=30)
    check("Concurrent heartbeat and API mutations raise no errors", not errors, errors[:3])

    # Deterministic guard for the race above: the tick must own the lock while
    # it touches the world model and resonant memory (deques also written by /observe).
    owned = {}
    orig_w, orig_m = main._WORLD_MODEL.observe, main._MEMORY.observe
    def w(*aa, **kk):
        owned["world"] = main._RUNTIME_LOCK._is_owned(); return orig_w(*aa, **kk)
    def m(*aa, **kk):
        owned["memory"] = main._RUNTIME_LOCK._is_owned(); return orig_m(*aa, **kk)
    main._WORLD_MODEL.observe, main._MEMORY.observe = w, m
    try:
        main._runtime_tick()
        client.post("/world-model/observe", json={"observation": {"v": 1.0}, "domain": "t", "persist": False})
        client.post("/memory/observe", json={"observation": {"v": 1.0}, "domain": "t", "persist": False})
    finally:
        main._WORLD_MODEL.observe, main._MEMORY.observe = orig_w, orig_m
    check("Heartbeat tick and standalone observe endpoints hold the runtime lock", owned.get("world") is True and owned.get("memory") is True, owned)
    check("All concurrent API calls succeed", all(s == 200 for s in statuses), statuses)
    finite = all(bool(torch.isfinite(p).all()) for p in main.get_model().parameters())
    check("Model parameters remain finite after concurrent training", finite)

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
