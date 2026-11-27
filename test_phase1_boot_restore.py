"""Phase 1 / B8: the corpus ingestor and shared optimizer are restored at boot.

A real process restart is simulated with two subprocesses sharing one state
directory: the first trains through the canonical corpus path plus the
language decoder and saves; the second boots cold and must come back with
the ingestor counters, the buffer, and the exact Adam moments, without any
checkpoint being quarantined.

Run with:
    python test_phase1_boot_restore.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

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


TEXTS = [
    "The river carries memory downstream and the delta remembers every flood it has known.",
    "Patterns repeat across scales; the fern and the coastline share one recursive grammar.",
    "A filter does not create the signal; it decides which part of it survives the passage.",
]


def _role(role: str) -> None:
    import torch
    from fastapi.testclient import TestClient

    import main

    def adam_fingerprint(opt):
        out = []
        for group in opt.param_groups:
            for param in group["params"]:
                st = opt.state.get(param) or {}
                if "exp_avg" in st:
                    out.append(round(float(st["exp_avg"].double().abs().sum()), 10))
                    step = st.get("step")
                    out.append(int(step.item()) if torch.is_tensor(step) else int(step or 0))
        return out

    with TestClient(main.app) as client:
        model = main.get_model()
        if role == "save":
            obs = [{"text": t, "entity_id": f"t{i}"} for i, t in enumerate(TEXTS)]
            for _ in range(3):
                main._train_z3_on_language_observations(obs, learning_rate=1e-3)
            client.post("/train-language", json={"text": TEXTS[0], "persist": False})
            client.post("/state/save")
            ing = main._LANGUAGE_INGESTOR
            result = {
                "trained_steps": ing.snapshot()["trained_steps"],
                "groups": len(main._OPTIMIZER.param_groups),
                "adam": adam_fingerprint(main._OPTIMIZER),
                "z3_norm": float(model.z3_state.norm()),
            }
        else:
            booted_with_ingestor = main._LANGUAGE_INGESTOR is not None
            runtime = client.get("/runtime").json()
            state = client.get("/state").json()
            ing = main._LANGUAGE_INGESTOR
            snap = ing.snapshot() if ing is not None else {}
            result = {
                "booted_with_ingestor": booted_with_ingestor,
                "runtime_has_ingestor": "language_ingestor" in runtime,
                "boot_restore": runtime.get("boot_restore", {}),
                "state_boot_restore": state.get("boot_restore", {}),
                "trained_steps": snap.get("trained_steps"),
                "health": snap.get("health_state"),
                "optimizer_restore": snap.get("optimizer_restore"),
                "groups": len(main._OPTIMIZER.param_groups) if main._OPTIMIZER is not None else 0,
                "adam": adam_fingerprint(main._OPTIMIZER) if main._OPTIMIZER is not None else [],
                "z3_norm": float(model.z3_state.norm()),
                "ingestor_shares_live_model": ing is not None and ing.model is model,
                "ingestor_shares_optimizer": ing is not None and ing.optimizer is main._OPTIMIZER,
            }
    print("RESULT " + json.dumps(result))


def _run(role: str, state_dir: str) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "Z3_STATE_DIR": state_dir,
            "DISABLE_REMOTE_CORPUS_STREAM": "1",
            "Z3_CORPUS_MIN_WORDS": "1",
            "LANGUAGE_TRAINING_MIN_WORDS": "1",
            "Z3_CORPUS_DEDUPLICATE_TEXTS": "false",
        }
    )
    env.pop("Z3_CORPUS_CHECKPOINT_PATH", None)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), role],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise RuntimeError(f"{role} subprocess failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")


def _optimizer_shape_change_unit() -> None:
    """A checkpoint written with an extra param group must restore partially, not be quarantined."""
    import torch

    from corpus_neural_ingestor import Z3CorpusIngestionConfig, Z3CorpusNeuralIngestor
    from Z3_neural_dynamics import Z3NeuralDynamics
    from z3_language_training import train_z3_on_language_window

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ingestor.pt")
        cfg = Z3CorpusIngestionConfig(enabled=True, checkpoint_path=path, min_words=1, deduplicate_texts=False)
        model = Z3NeuralDynamics()
        extra = torch.nn.Linear(4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.add_param_group({"params": list(extra.parameters()), "lr": 1e-3})
        ing = Z3CorpusNeuralIngestor(cfg, model=model, optimizer=opt, trainer=train_z3_on_language_window, own_model=False)
        ing.train_texts_now(TEXTS)
        check("Unit: ingestor trained before save", ing.snapshot()["trained_steps"] >= 1, ing.snapshot())
        check("Unit: checkpoint saved", ing.save_checkpoint())
        first = next(iter(model.parameters()))
        saved_moment = opt.state[first]["exp_avg"].clone()

        fresh_model = Z3NeuralDynamics()
        fresh_model.load_state_dict(model.state_dict())
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)  # one group only
        restored = Z3CorpusNeuralIngestor(cfg, model=fresh_model, optimizer=fresh_opt, trainer=train_z3_on_language_window, own_model=False)
        snap = restored.snapshot()
        check("Unit: shape change is a partial restore, not corruption", str(snap.get("optimizer_restore", "")).startswith("partial:1/2"), snap.get("optimizer_restore"))
        check("Unit: checkpoint is not quarantined", not [f for f in os.listdir(tmp) if ".corrupt." in f], os.listdir(tmp))
        check("Unit: ingestor counters restored", snap.get("trained_steps") == ing.snapshot()["trained_steps"], snap.get("trained_steps"))
        fresh_first = next(iter(fresh_model.parameters()))
        check(
            "Unit: Z³ Adam moments restored exactly",
            bool(torch.equal(fresh_opt.state[fresh_first]["exp_avg"], saved_moment)),
        )
        # A partial restore genuinely dropped some optimizer history: it must be
        # surfaced as a warning (not hidden as "ready", not escalated to degraded).
        check("Unit: partial restore is surfaced as a warning, not degradation", snap.get("health_state") == "warning" and "optimizer_restore_partial" in str(snap.get("last_warning", "")), (snap.get("health_state"), snap.get("last_warning")))


if __name__ == "__main__" and len(sys.argv) > 1:
    _role(sys.argv[1])
    raise SystemExit(0)

print("=" * 60)
print("TESTING PHASE 1 BOOT RESTORE")
print("=" * 60)

with tempfile.TemporaryDirectory(prefix="z3_boot_") as state_dir:
    saved = _run("save", state_dir)
    booted = _run("boot", state_dir)
    files = os.listdir(state_dir)

check("Saved run trained the ingestor", saved["trained_steps"] >= 1, saved)
check("Ingestor exists immediately after a cold boot", booted["booted_with_ingestor"], booted)
check("/runtime reports the ingestor after boot", booted["runtime_has_ingestor"], booted)
check("Ingestor counters survive the restart", booted["trained_steps"] == saved["trained_steps"], (saved["trained_steps"], booted["trained_steps"]))
check("Ingestor health is ready after boot", booted["health"] == "ready", booted["health"])
check("Optimizer state restored fully (same shape every boot)", booted["optimizer_restore"] == "full", booted["optimizer_restore"])
check("Optimizer param groups match across restart", booted["groups"] == saved["groups"] == 2, (saved["groups"], booted["groups"]))
check("Adam moments and step counts are identical after restart", booted["adam"] == saved["adam"] and len(saved["adam"]) > 0, (len(saved["adam"]), len(booted["adam"])))
check("Recurrent Z³ state survives the restart", abs(booted["z3_norm"] - saved["z3_norm"]) < 1e-6, (saved["z3_norm"], booted["z3_norm"]))
check("Ingestor is bound to the live model", booted["ingestor_shares_live_model"], booted)
check("Ingestor is bound to the shared optimizer", booted["ingestor_shares_optimizer"], booted)
check("Boot restore report is surfaced on /runtime", booted["boot_restore"].get("corpus_ingestor", {}).get("restored") is True, booted["boot_restore"])
check("Boot restore report is surfaced on /state", booted["state_boot_restore"].get("loaded", {}).get("corpus_ingestor") is True, booted["state_boot_restore"])
check("No checkpoint was quarantined", not [f for f in files if ".corrupt." in f], files)

_optimizer_shape_change_unit()

print("=" * 60)
print(f"PASS={PASS} FAIL={FAIL}")
raise SystemExit(1 if FAIL else 0)
