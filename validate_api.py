"""Local validation for the Railway FastAPI membrane."""
from __future__ import annotations

import json
import tomllib

from fastapi.testclient import TestClient

import main

with open("pyproject.toml", "rb") as handle:
    tomllib.load(handle)

with open("railway.json", "r", encoding="utf-8") as handle:
    data = json.load(handle)

assert data["deploy"]["startCommand"] == "python main.py"

client = TestClient(main.app)

root = client.get("/")
assert root.status_code == 200, root.text
assert "Z³ Neural Runtime" in root.text

api = client.get("/api")
assert api.status_code == 200, api.text
assert api.json().get("interface") == "/interface"
assert api.json().get("world_model") == "/world-model"
assert api.json().get("memory") == "/memory"
assert api.json().get("runtime") == "/runtime"
assert api.json().get("language") == "/language"
assert api.json().get("infra") == "/infra"
assert api.json().get("chat") == "/chat"

interface = client.get("/interface")
assert interface.status_code == 200, interface.text
assert "Integrated Observe" in interface.text
assert "Autonomous Runtime" in interface.text
assert "Language Training Stream" in interface.text
assert "Infrastructure" in interface.text
assert "Language auto-ingest" in interface.text

health = client.get("/health")
assert health.status_code == 200, health.text

config = client.get("/config")
assert config.status_code == 200, config.text
assert "world_model" in config.json()
assert "memory" in config.json()

step = client.post("/step", json={"x": [0.0] * 16})
assert step.status_code == 200, step.text

train_step = client.post("/train-step", json={"x": [0.0] * 16, "persist": False})
assert train_step.status_code == 200, train_step.text

observation = {
    "content": "validation observation",
    "tone": 0.42,
    "salience": 0.7,
    "symbol": "Z3",
}
world = client.post("/world-model/observe", json={"observation": observation, "domain": "validation", "persist": False})
assert world.status_code == 200, world.text
assert "world_model" in world.json()

memory = client.post("/memory/observe", json={"observation": observation, "domain": "validation", "persist": False})
assert memory.status_code == 200, memory.text
assert "memory" in memory.json()

observe = client.post("/observe", json={"observation": observation, "domain": "validation", "train": False, "persist": False})
assert observe.status_code == 200, observe.text
assert "world_model" in observe.json()
assert "memory" in observe.json()
assert "z3" in observe.json()
assert len(observe.json()["input_vector"]) == 16

world_state = client.get("/world-model")
assert world_state.status_code == 200, world_state.text
memory_state = client.get("/memory")
assert memory_state.status_code == 200, memory_state.text
state = client.get("/state")
assert state.status_code == 200, state.text
runtime = client.get("/runtime")
assert runtime.status_code == 200, runtime.text
language = client.get("/language")
assert language.status_code == 200, language.text
assert language.json().get("dataset", {}).get("domain") == "language:corpus"
assert "running" in runtime.json()
assert "language_schedule" in runtime.json()
assert runtime.json()["language_schedule"].get("enabled") is True
infra = client.get("/infra")
assert infra.status_code == 200, infra.text
assert "volume" in infra.json()
assert "qdrant" in infra.json()
assert "redis" in infra.json()
assert "postgres" in infra.json()
infra_sync = client.post("/infra/sync")
assert infra_sync.status_code == 200, infra_sync.text
runtime_tick = client.post("/runtime/tick")
assert runtime_tick.status_code == 200, runtime_tick.text
assert runtime_tick.json().get("tick_count", 0) >= 1
runtime_start = client.post(
    "/runtime/start",
    json={
        "interval_seconds": 60,
        "autosave_every_ticks": 2,
        "language_enabled": True,
        "language_every_ticks": 3,
        "language_batch_size": 1,
        "language_train": False,
        "language_learning_rate": 0.001,
    },
)
assert runtime_start.status_code == 200, runtime_start.text
assert runtime_start.json()["language_schedule"]["every_ticks"] == 3
assert runtime_start.json()["language_schedule"]["batch_size"] == 1
runtime_stop = client.post("/runtime/stop")
assert runtime_stop.status_code == 200, runtime_stop.text
save = client.post("/state/save")
assert save.status_code == 200, save.text
load = client.post("/state/load")
assert load.status_code == 200, load.text

print("api validation ok")
