import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.spatial_agent import agent, tools
from api.routes import SettingsUpdateRequest


class ModelProviderConfigTests(unittest.TestCase):
    def test_settings_schema_accepts_independent_provider_configuration(self):
        settings = SettingsUpdateRequest(
            camera_host="camera.example",
            camera_user="admin",
            chat_base_url="https://chat.example/v1",
            chat_api_key="chat-secret",
            chat_model="chat-model",
            vlm_base_url="https://vision.example/v1",
            vlm_api_key="vision-secret",
            vlm_model="vision-model",
        )

        self.assertEqual("https://chat.example/v1", settings.chat_base_url)
        self.assertEqual("chat-secret", settings.chat_api_key)
        self.assertEqual("https://vision.example/v1", settings.vlm_base_url)
        self.assertEqual("vision-secret", settings.vlm_api_key)

    def test_chat_agent_uses_only_chat_provider_settings(self):
        captured = {}

        def fake_chat_openai(**kwargs):
            captured.update(kwargs)
            return object()

        environment = {
            "CHAT_BASE_URL": "https://chat.example/v1",
            "CHAT_API_KEY": "chat-secret",
            "CHAT_MODEL": "chat-model",
            "VLM_BASE_URL": "https://vision.example/v1",
            "VLM_API_KEY": "vision-secret",
            "VLM_MODEL": "vision-model",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(agent, "ChatOpenAI", fake_chat_openai):
            agent.create_agent_model()

        self.assertEqual("https://chat.example/v1", captured["base_url"])
        self.assertEqual("chat-secret", captured["api_key"])
        self.assertEqual("chat-model", captured["model"])

    def test_vlm_uses_only_vision_provider_settings(self):
        captured = {}

        def fake_chat_openai(**kwargs):
            captured.update(kwargs)
            return object()

        environment = {
            "CHAT_BASE_URL": "https://chat.example/v1",
            "CHAT_API_KEY": "chat-secret",
            "CHAT_MODEL": "chat-model",
            "VLM_BASE_URL": "https://vision.example/v1",
            "VLM_API_KEY": "vision-secret",
            "VLM_MODEL": "vision-model",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(tools, "ChatOpenAI", fake_chat_openai):
            tools._get_vlm()

        self.assertEqual("https://vision.example/v1", captured["base_url"])
        self.assertEqual("vision-secret", captured["api_key"])
        self.assertEqual("vision-model", captured["model"])


if __name__ == "__main__":
    unittest.main()
