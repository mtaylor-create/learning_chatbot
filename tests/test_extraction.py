"""Tests for extraction.py — fact extraction from conversation turns."""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extraction import _parse_facts, extract_facts


class TestParseFacts:
    def test_returns_empty_for_none_response(self):
        assert _parse_facts("NONE") == []
        assert _parse_facts("none") == []
        assert _parse_facts("None") == []
        assert _parse_facts("NONE\n") == []

    def test_returns_empty_for_empty_string(self):
        assert _parse_facts("") == []
        assert _parse_facts("   ") == []

    def test_extracts_plain_lines(self):
        raw = "The user's name is Jordan.\nThe user likes hiking."
        facts = _parse_facts(raw)
        assert len(facts) == 2
        assert facts[0] == "The user's name is Jordan."
        assert facts[1] == "The user likes hiking."

    def test_strips_bullet_points(self):
        raw = "- The user has a cat.\n* The user likes tea.\n• The user is 30."
        facts = _parse_facts(raw)
        assert len(facts) == 3
        assert facts[0] == "The user has a cat."

    def test_strips_numbering(self):
        raw = "1. The user works at Google.\n2) The user plays guitar."
        facts = _parse_facts(raw)
        assert len(facts) == 2
        assert facts[0] == "The user works at Google."

    def test_filters_short_lines(self):
        raw = "The user likes hiking.\nOk\nYes\n\nThe user is from Texas."
        facts = _parse_facts(raw)
        assert len(facts) == 2
        assert all(len(f) >= 6 for f in facts)


class TestExtractFacts:
    @patch("extraction.generate")
    def test_calls_generate_and_parses(self, mock_generate):
        mock_generate.return_value = "The user's name is Alex.\nThe user likes cats."

        facts = extract_facts("mistral", "My name is Alex and I love cats.", "Nice to meet you Alex!")

        assert len(facts) == 2
        assert "Alex" in facts[0]
        mock_generate.assert_called_once()

    @patch("extraction.generate")
    def test_returns_empty_on_none(self, mock_generate):
        mock_generate.return_value = "NONE"
        facts = extract_facts("mistral", "How's the weather?", "I'm not sure!")
        assert facts == []
