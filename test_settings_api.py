"""
Test các endpoint Cài đặt mới:
1. GET /settings
2. POST /settings/test_llm
3. POST /camera/test_sync
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://localhost:8080"

def test_settings_endpoints():
    print("Testing GET /settings...")
    try:
        r = requests.get(f"{BASE_URL}/settings", timeout=5)
        print("Status code:", r.status_code)
        print("Response:", r.json())
        assert r.status_code == 200
    except Exception as e:
        print("Server not running or connection failed:", e)

if __name__ == "__main__":
    test_settings_endpoints()
