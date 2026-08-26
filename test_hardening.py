"""Regression checks for persistence recovery and API security controls."""
from __future__ import annotations

from tempfile import TemporaryDirectory

from state_store import StateStore


class DummyModel:
    def save_checkpoint(self, path):
        path.write_text("dummy", encoding="utf-8")


class DummyModelClass:
    @staticmethod
    def load_checkpoint(path, map_location="cpu"):
        return DummyModel()


class DummyState:
    def __init__(self):
        self.loaded = None

    def save_state(self):
        return {"value": "healthy"}

    def load_state(self, state):
        self.loaded = state

    def export_state(self):
        return {"value": "healthy"}


with TemporaryDirectory() as temp_dir:
    store = StateStore(temp_dir)
    world = DummyState()
    memory = DummyState()
    store.save_all(model=DummyModel(), world_model=world, memory=memory)
    store.save_all(model=DummyModel(), world_model=world, memory=memory)
    store.world_model_path.write_text("{broken", encoding="utf-8")
    loaded = store.load_all(model=None, model_cls=DummyModelClass, world_model=world, memory=memory)
    assert loaded["recovered"].get("world_model"), loaded
    assert world.loaded == {"value": "healthy"}, world.loaded
    assert list(store.state_dir.glob("world_model.json.corrupt.*")), "corrupt state was not quarantined"

import main
from fastapi.testclient import TestClient

previous_required = main._AUTH_REQUIRED
previous_token = main._API_TOKEN
try:
    main._AUTH_REQUIRED = True
    main._API_TOKEN = "test-token"
    with TestClient(main.app) as client:
        assert client.get("/runtime").status_code == 401
        authorized = client.get("/runtime", headers={"Authorization": "Bearer test-token"})
        assert authorized.status_code == 200, authorized.text
finally:
    main._AUTH_REQUIRED = previous_required
    main._API_TOKEN = previous_token

print("hardening checks passed")
