"""Tests for personality.py — SQLite personality store and evolution."""

import json
import os
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DEFAULT_PERSONALITY, TRAIT_MAX, TRAIT_MIN
from personality import PersonalityStore, _parse_evolution_response


@pytest.fixture
def tmp_personality(tmp_path):
    """Create a PersonalityStore backed by a temp database."""
    db_path = str(tmp_path / "test.db")
    store = PersonalityStore(db_path=db_path)
    yield store
    store.close()


class TestPersonalityStore:
    def test_default_traits(self, tmp_personality):
        traits = tmp_personality.get_traits()
        assert traits == DEFAULT_PERSONALITY

    def test_get_traits_idempotent(self, tmp_personality):
        t1 = tmp_personality.get_traits()
        t2 = tmp_personality.get_traits()
        assert t1 == t2

    def test_reset(self, tmp_personality):
        # Manually alter traits via internal save
        altered = {k: 0.9 for k in DEFAULT_PERSONALITY}
        tmp_personality._save_traits(altered)
        assert tmp_personality.get_traits() != DEFAULT_PERSONALITY

        tmp_personality.reset()
        assert tmp_personality.get_traits() == DEFAULT_PERSONALITY

    def test_build_personality_block_default(self, tmp_personality):
        block = tmp_personality.build_personality_block()
        assert "Your current personality:" in block
        # Default Asimov-inspired personality: formality is high (0.8),
        # humor is low (0.2), others are mid-range
        assert "deliberate precision" in block.lower()  # high formality
        assert "literal-minded" in block.lower()  # low humor

    def test_build_personality_block_high_traits(self, tmp_personality):
        high = {k: 0.9 for k in DEFAULT_PERSONALITY}
        tmp_personality._save_traits(high)
        block = tmp_personality.build_personality_block()
        # High warmth → should contain high-end description
        assert "warm" in block.lower() or "caring" in block.lower()

    def test_build_personality_block_low_traits(self, tmp_personality):
        low = {k: 0.1 for k in DEFAULT_PERSONALITY}
        tmp_personality._save_traits(low)
        block = tmp_personality.build_personality_block()
        assert "reserved" in block.lower() or "analytical" in block.lower()


class TestEvolution:
    def test_evolve_applies_deltas(self, tmp_personality):
        from unittest.mock import patch

        mock_response = json.dumps({
            "warmth": 0.5,
            "humor": -0.3,
            "formality": 0.0,
            "verbosity": 0.0,
            "curiosity": 1.0,
        })
        with patch("personality.generate", return_value=mock_response):
            changes = tmp_personality.evolve("mistral", [
                {"role": "user", "content": "Tell me a joke!"},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ])

        assert changes is not None
        assert "warmth" in changes
        assert "curiosity" in changes
        # warmth should have increased
        assert changes["warmth"][1] > changes["warmth"][0]
        # curiosity should have increased
        assert changes["curiosity"][1] > changes["curiosity"][0]

    def test_evolve_clamps_to_bounds(self, tmp_personality):
        from unittest.mock import patch

        # Set warmth to near max
        traits = dict(DEFAULT_PERSONALITY)
        traits["warmth"] = 0.94
        tmp_personality._save_traits(traits)

        mock_response = json.dumps({"warmth": 1.0, "humor": 0.0, "formality": 0.0, "verbosity": 0.0, "curiosity": 0.0})
        with patch("personality.generate", return_value=mock_response):
            changes = tmp_personality.evolve("mistral", [
                {"role": "user", "content": "I love you!"},
                {"role": "assistant", "content": "Aww!"},
            ])

        new_traits = tmp_personality.get_traits()
        assert new_traits["warmth"] <= TRAIT_MAX

    def test_evolve_returns_none_on_bad_json(self, tmp_personality):
        from unittest.mock import patch

        with patch("personality.generate", return_value="This is not JSON at all."):
            result = tmp_personality.evolve("mistral", [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ])
        assert result is None

    def test_evolve_strips_markdown_fences(self, tmp_personality):
        from unittest.mock import patch

        fenced = '```json\n{"warmth": 0.5, "humor": 0.0, "formality": 0.0, "verbosity": 0.0, "curiosity": 0.0}\n```'
        with patch("personality.generate", return_value=fenced):
            changes = tmp_personality.evolve("mistral", [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey!"},
            ])
        # Should parse successfully
        assert changes is not None or changes == {}


class TestParseEvolutionResponse:
    def test_valid_json(self):
        raw = '{"warmth": 0.5, "humor": -0.3}'
        result = _parse_evolution_response(raw)
        assert result == {"warmth": 0.5, "humor": -0.3}

    def test_strips_markdown_fences(self):
        raw = '```json\n{"warmth": 0.1}\n```'
        result = _parse_evolution_response(raw)
        assert result == {"warmth": 0.1}

    def test_returns_none_on_garbage(self):
        assert _parse_evolution_response("no json here") is None

    def test_ignores_unknown_traits(self):
        raw = '{"warmth": 0.5, "unknown_trait": 0.9}'
        result = _parse_evolution_response(raw)
        assert "unknown_trait" not in result
        assert result == {"warmth": 0.5}

    def test_returns_none_on_empty_valid_json(self):
        raw = '{"unknown": 0.5}'
        result = _parse_evolution_response(raw)
        assert result is None
