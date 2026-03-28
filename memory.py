"""ChromaDB long-term memory store with semantic deduplication."""

import os
import time
import uuid

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, DEDUP_THRESHOLD, MEMORY_COLLECTION, MEMORY_TOP_K


class MemoryStore:
    """Stores fact embeddings, retrieves relevant memories by similarity."""

    def __init__(self, persist_dir=CHROMA_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_memories(self, facts):
        """Store facts after deduplication. Return count of facts actually added."""
        added = 0
        for fact in facts:
            fact = fact.strip()
            if not fact:
                continue
            if self._is_duplicate(fact):
                continue
            self._collection.add(
                ids=[str(uuid.uuid4())],
                documents=[fact],
                metadatas=[{"timestamp": time.time()}],
            )
            added += 1
        return added

    def retrieve(self, query, top_k=MEMORY_TOP_K):
        """Return the top_k most relevant memories for the given query."""
        total = self._collection.count()
        if total == 0:
            return []
        k = min(top_k, total)
        results = self._collection.query(query_texts=[query], n_results=k)
        return results["documents"][0] if results["documents"] else []

    def retrieve_with_ids(self, query, top_k=MEMORY_TOP_K):
        """Return the top_k most relevant memories as (id, document) tuples."""
        total = self._collection.count()
        if total == 0:
            return []
        k = min(top_k, total)
        results = self._collection.query(query_texts=[query], n_results=k)
        if not results["documents"] or not results["documents"][0]:
            return []
        ids = results["ids"][0]
        docs = results["documents"][0]
        return list(zip(ids, docs))

    def delete_memory(self, memory_id):
        """Delete a single memory by its ChromaDB ID."""
        self._collection.delete(ids=[memory_id])

    def update_memory(self, memory_id, new_text):
        """Replace the document text of a memory, preserving its ID."""
        self._collection.update(
            ids=[memory_id],
            documents=[new_text],
            metadatas=[{"timestamp": time.time()}],
        )

    def count(self):
        """Return the total number of stored memories."""
        return self._collection.count()

    def clear(self):
        """Delete and recreate the collection."""
        self._client.delete_collection(MEMORY_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _is_duplicate(self, fact):
        """Check if a near-duplicate of this fact already exists."""
        if self._collection.count() == 0:
            return False
        results = self._collection.query(query_texts=[fact], n_results=1)
        if not results["distances"] or not results["distances"][0]:
            return False
        distance = results["distances"][0][0]
        similarity = 1 - distance
        return similarity >= DEDUP_THRESHOLD
