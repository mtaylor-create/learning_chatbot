"""Tests for llm.py — Ollama API wrapper."""

import json
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm import chat_completion, check_ollama, generate


class TestChatCompletion:
    @patch("llm.requests.post")
    def test_returns_assistant_content(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Hello there!"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = chat_completion("mistral", "You are a bot.", [{"role": "user", "content": "Hi"}])

        assert result == "Hello there!"
        call_json = mock_post.call_args[1]["json"]
        assert call_json["model"] == "mistral"
        assert call_json["stream"] is False
        assert call_json["messages"][0]["role"] == "system"
        assert call_json["messages"][0]["content"] == "You are a bot."
        assert call_json["messages"][1]["role"] == "user"

    @patch("llm.requests.post")
    def test_uses_120s_timeout(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        chat_completion("mistral", "sys", [])
        assert mock_post.call_args[1]["timeout"] == 120

    @patch("llm.requests.post")
    def test_custom_temperature(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        chat_completion("mistral", "sys", [], temperature=0.2)
        call_json = mock_post.call_args[1]["json"]
        assert call_json["options"]["temperature"] == 0.2


class TestGenerate:
    @patch("llm.requests.post")
    def test_returns_response_field(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Extracted fact."}
        mock_post.return_value = mock_resp

        result = generate("mistral", "Extract facts from this.")
        assert result == "Extracted fact."

    @patch("llm.requests.post")
    def test_uses_low_temperature_by_default(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "ok"}
        mock_post.return_value = mock_resp

        generate("mistral", "prompt")
        call_json = mock_post.call_args[1]["json"]
        assert call_json["options"]["temperature"] == 0.3


class TestCheckOllama:
    @patch("llm.requests.get")
    def test_returns_true_when_model_available(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "mistral:latest"}]}
        mock_get.return_value = mock_resp

        assert check_ollama("mistral") is True

    @patch("llm.requests.get")
    def test_returns_true_for_exact_match(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "mistral"}]}
        mock_get.return_value = mock_resp

        assert check_ollama("mistral") is True

    @patch("llm.requests.get")
    def test_returns_false_when_model_missing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "llama2:latest"}]}
        mock_get.return_value = mock_resp

        assert check_ollama("mistral") is False

    @patch("llm.requests.get")
    def test_returns_false_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("refused")
        assert check_ollama("mistral") is False

    @patch("llm.requests.get")
    def test_returns_false_on_timeout(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        assert check_ollama("mistral") is False
