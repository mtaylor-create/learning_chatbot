"""Tests for memory.py — ChromaDB long-term memory store."""

import os
import shutil
import tempfile

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory import MemoryStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a MemoryStore backed by a temp directory."""
    store = MemoryStore(persist_dir=str(tmp_path / "chroma"))
    yield store


class TestMemoryStore:
    def test_initially_empty(self, tmp_store):
        assert tmp_store.count() == 0

    def test_add_and_count(self, tmp_store):
        added = tmp_store.add_memories(["The user likes pizza.", "The user has a dog named Rex."])
        assert added == 2
        assert tmp_store.count() == 2

    def test_retrieve_returns_relevant(self, tmp_store):
        tmp_store.add_memories([
            "The user's favorite color is blue.",
            "The user works as a nurse.",
            "The user has two cats.",
        ])
        results = tmp_store.retrieve("What pets does the user have?", top_k=2)
        assert len(results) == 2
        # The cat fact should be among the top results
        assert any("cats" in r.lower() for r in results)

    def test_retrieve_empty_collection(self, tmp_store):
        results = tmp_store.retrieve("anything")
        assert results == []

    def test_retrieve_clamps_top_k(self, tmp_store):
        tmp_store.add_memories(["Fact one.", "Fact two."])
        results = tmp_store.retrieve("something", top_k=10)
        assert len(results) == 2

    def test_deduplication(self, tmp_store):
        tmp_store.add_memories(["The user's name is Jordan."])
        # Near-duplicate should be rejected
        added = tmp_store.add_memories(["The user's name is Jordan."])
        assert added == 0
        assert tmp_store.count() == 1

    def test_different_facts_not_deduped(self, tmp_store):
        tmp_store.add_memories(["The user likes pizza."])
        added = tmp_store.add_memories(["The user works as an engineer."])
        assert added == 1
        assert tmp_store.count() == 2

    def test_clear(self, tmp_store):
        tmp_store.add_memories(["Some fact."])
        assert tmp_store.count() == 1
        tmp_store.clear()
        assert tmp_store.count() == 0

    def test_skips_empty_strings(self, tmp_store):
        added = tmp_store.add_memories(["", "  ", "Valid fact."])
        assert added == 1
        assert tmp_store.count() == 1
