"""Semantic retrieval over menu items with ChromaDB, plus a keyword baseline used in evaluation."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from .models import MenuItem

ScoredItem = Tuple[MenuItem, float]


class SemanticRetriever:
    """Embeds every menu item once, then finds the closest items to a request.

    Uses Chroma's default embedding model (all-MiniLM-L6-v2, run locally through ONNX),
    so no API key is needed for retrieval. Similarity is cosine similarity in [-1, 1].
    """

    def __init__(self, items: List[MenuItem], collection_name: str = "menu"):
        self._items: Dict[str, MenuItem] = {i.item_id: i for i in items}
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.upsert(
            ids=[i.item_id for i in items],
            documents=[i.document() for i in items],
            metadatas=[{"restaurant": i.restaurant, "cuisine": i.cuisine, "price": i.price} for i in items],
        )

    def search(self, query: str, k: int = 15, allowed_ids: Iterable[str] | None = None) -> List[ScoredItem]:
        """Top-k items for the query, restricted to allowed_ids when given."""
        allowed = list(allowed_ids) if allowed_ids is not None else list(self._items)
        if not allowed:
            return []
        restrict = len(allowed) < len(self._items)
        result = self._collection.query(
            query_texts=[query],
            n_results=min(k, len(allowed)),
            ids=allowed if restrict else None,  # search only items that passed the hard constraints
        )
        ids = result["ids"][0]
        distances = result["distances"][0]
        return [(self._items[i], 1.0 - d) for i, d in zip(ids, distances)]


_TOKEN = re.compile(r"[a-z]+")
_STOP = {"a", "an", "the", "and", "or", "for", "to", "of", "with", "something", "some", "i", "want", "me",
         "please", "today", "tonight", "would", "like", "that", "is", "it", "on", "in", "my", "get", "give",
         "under", "dollars", "not", "too", "very", "any", "food", "meal", "eat", "craving", "kind"}


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class KeywordRetriever:
    """Baseline: rank items by how many request words appear in the dish text (no embeddings)."""

    def __init__(self, items: List[MenuItem]):
        self._items = items
        self._docs = {i.item_id: Counter(_tokens(i.document())) for i in items}

    def search(self, query: str, k: int = 15, allowed_ids: Iterable[str] | None = None) -> List[ScoredItem]:
        allowed = set(allowed_ids) if allowed_ids is not None else {i.item_id for i in self._items}
        q = set(_tokens(query))
        scored = [(i, float(sum(1 for t in q if t in self._docs[i.item_id])))
                  for i in self._items if i.item_id in allowed]
        scored.sort(key=lambda pair: (-pair[1], pair[0].item_id))
        return scored[:k]
