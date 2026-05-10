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
assert root.status_code == 200
assert root.json().get("interface") == "/interface"
interface = client.get("/interface")
assert interface.status_code == 200, interface.text
assert "Z³ Neural Network Interface" in interface.text
health = client.get("/health")
assert health.status_code == 200, health.text
config = client.get("/config")
assert config.status_code == 200, config.text
step = client.post("/step", json={"x": [0.0] * 16})
assert step.status_code == 200, step.text

print("api validation ok")
