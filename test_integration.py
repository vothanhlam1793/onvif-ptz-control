"""
Integration test: khởi động FastAPI app ngầm và test các endpoint bao gồm Virtual PTZ.
"""
import time
from fastapi.testclient import TestClient
from main import app
import api.routes as routes_module

client = TestClient(app)

with client:
    # Test Root
    res = client.get("/")
    assert res.status_code == 200
    print("GET / OK")

    # Test Status
    res = client.get("/ptz/status")
    assert res.status_code == 200
    data = res.json()
    print(f"GET /ptz/status OK: {data}")
    assert "virtual_pan" in data

    # Test Virtual Goto
    res = client.post("/ptz/virtual/goto", json={"pan": 0.0, "tilt": 0.0, "speed": 0.8})
    assert res.status_code == 200
    print(f"POST /ptz/virtual/goto OK: {res.json()}")

    # Test Presets
    res = client.get("/ptz/presets")
    assert res.status_code == 200
    print(f"GET /ptz/presets OK: {res.json()}")

print("ALL APP & VIRTUAL PTZ INTEGRATION TESTS PASSED!")
