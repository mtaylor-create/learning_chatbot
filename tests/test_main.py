"""Tests for main.py — system prompt construction and slash commands."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import build_system_prompt, handle_slash_command


class TestBuildSystemPrompt:
    def test_includes_personality_and_memory(self):
        personality = MagicMock()
        personality.build_personality_block.return_value = "Your current personality:\n- Warm and caring."

        memory = MagicMock()
        memory.retrieve_with_ids.return_value = [
            ("id1", "The user likes cats."),
            ("id2", "The user's name is Jordan."),
        ]

        prompt, fetched = build_system_prompt(personality, memory, "Tell me about cats")

        assert "Asimov" in prompt
        assert "Warm and caring" in prompt
        assert "The user likes cats." in prompt
        assert "The user's name is Jordan." in prompt
        assert "Things you know about the user" in prompt
        assert len(fetched) == 2

    def test_empty_memory_block(self):
        personality = MagicMock()
        personality.build_personality_block.return_value = "Your current personality:\n- Balanced."

        memory = MagicMock()
        memory.retrieve_with_ids.return_value = []

        prompt, fetched = build_system_prompt(personality, memory, "Hello")

        assert "don't know anything about the user yet" in prompt
        assert fetched == []

    def test_contains_guidelines(self):
        personality = MagicMock()
        personality.build_personality_block.return_value = "Block"
        memory = MagicMock()
        memory.retrieve_with_ids.return_value = []

        prompt, _ = build_system_prompt(personality, memory, "Hi")
        assert "Guidelines:" in prompt
        assert "Speak naturally" in prompt


class TestSlashCommands:
    def test_traits_command(self, capsys):
        personality = MagicMock()
        personality.get_traits.return_value = {
            "warmth": 0.50,
            "humor": 0.80,
            "formality": 0.20,
            "verbosity": 0.50,
            "curiosity": 0.65,
        }
        memory = MagicMock()

        result = handle_slash_command("/traits", personality, memory, [])
        assert result is True

        output = capsys.readouterr().out
        assert "warmth" in output
        assert "humor" in output
        assert "0.50" in output
        assert "0.80" in output

    def test_memories_command(self, capsys):
        personality = MagicMock()
        memory = MagicMock()
        memory.count.return_value = 42

        result = handle_slash_command("/memories", personality, memory, [])
        assert result is True

        output = capsys.readouterr().out
        assert "42" in output

    def test_unknown_command(self):
        result = handle_slash_command("/unknown", MagicMock(), MagicMock(), [])
        assert result is False
