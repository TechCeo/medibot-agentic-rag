from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from src import config
from src.orchestrator import MediBotReActOrchestrator, ReActRunResult
from src.vector_service import normalize_query


DEFAULT_CACHE_DB = config.PROJECT_ROOT / "cache" / "semantic_cache.sqlite3"
DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass(frozen=True)
class CacheLookupResult:
    hit: bool
    similarity: float
    query: str | None
    answer: str | None
    trace: list[dict[str, Any]]
    latency_ms: float


@dataclass(frozen=True)
class CachedOrchestratorResult:
    answer: str
    trace: list[dict[str, Any]]
    cache_hit: bool
    similarity: float
    latency_ms: float
    source_query: str | None


class SemanticCache:
    """SQLite-backed semantic response cache for MediBot orchestrator outputs."""

    def __init__(
        self,
        db_path: Path = DEFAULT_CACHE_DB,
        vectorizer_path: Path = config.VECTORIZER_FILE,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.db_path = db_path
        self.vectorizer_path = vectorizer_path
        self.similarity_threshold = similarity_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorizer = self._load_vectorizer()
        self._init_db()

    def _load_vectorizer(self) -> Any:
        if not self.vectorizer_path.exists():
            raise FileNotFoundError(
                f"Missing vectorizer at {self.vectorizer_path}. Run scripts/build_index.py first."
            )
        with self.vectorizer_path.open("rb") as handle:
            return pickle.load(handle)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_cache_created_at ON semantic_cache(created_at)"
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM semantic_cache")

    def embed_query(self, query: str) -> np.ndarray:
        normalized_query_text = normalize_query(query)
        sparse = self.vectorizer.transform([normalized_query_text])
        dense = sparse.astype(np.float32).toarray()
        return normalize(dense, norm="l2").astype(np.float32)[0]

    def lookup(self, query: str) -> CacheLookupResult:
        start = time.perf_counter()
        query_vector = self.embed_query(query)
        best_similarity = -1.0
        best_row: tuple[str, str, str] | None = None

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT query, answer, trace_json, vector_json FROM semantic_cache"
            ).fetchall()

        for cached_query, answer, trace_json, vector_json in rows:
            cached_vector = np.array(json.loads(vector_json), dtype=np.float32)
            similarity = float(np.dot(query_vector, cached_vector))
            if similarity > best_similarity:
                best_similarity = similarity
                best_row = (cached_query, answer, trace_json)

        latency_ms = (time.perf_counter() - start) * 1000
        if best_row and best_similarity >= self.similarity_threshold:
            cached_query, answer, trace_json = best_row
            return CacheLookupResult(
                hit=True,
                similarity=round(best_similarity, 4),
                query=cached_query,
                answer=answer,
                trace=json.loads(trace_json),
                latency_ms=round(latency_ms, 3),
            )
        return CacheLookupResult(
            hit=False,
            similarity=round(max(best_similarity, 0.0), 4),
            query=best_row[0] if best_row else None,
            answer=None,
            trace=[],
            latency_ms=round(latency_ms, 3),
        )

    def store(self, query: str, answer: str, trace: list[dict[str, Any]]) -> None:
        vector = self.embed_query(query)
        normalized_query_text = normalize_query(query)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache
                    (query, normalized_query, answer, trace_json, vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    normalized_query_text,
                    answer,
                    json.dumps(trace, ensure_ascii=False),
                    json.dumps(vector.tolist()),
                    time.time(),
                ),
            )


class CachedMediBotOrchestrator:
    """Thin semantic-cache wrapper around MediBotReActOrchestrator."""

    def __init__(
        self,
        orchestrator: MediBotReActOrchestrator | None = None,
        cache: SemanticCache | None = None,
    ) -> None:
        self.orchestrator = orchestrator or MediBotReActOrchestrator()
        self.cache = cache or SemanticCache()

    def invoke(self, user_message: str, session_id: str = "default") -> CachedOrchestratorResult:
        start = time.perf_counter()
        lookup = self.cache.lookup(user_message)
        if lookup.hit and lookup.answer is not None:
            return CachedOrchestratorResult(
                answer=lookup.answer,
                trace=[
                    {
                        "type": "cache_hit",
                        "similarity": lookup.similarity,
                        "source_query": lookup.query,
                    },
                    *lookup.trace,
                ],
                cache_hit=True,
                similarity=lookup.similarity,
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
                source_query=lookup.query,
            )

        result: ReActRunResult = self.orchestrator.invoke(user_message, session_id=session_id)
        self.cache.store(user_message, result.final_answer, result.trace)
        return CachedOrchestratorResult(
            answer=result.final_answer,
            trace=result.trace,
            cache_hit=False,
            similarity=lookup.similarity,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            source_query=lookup.query,
        )
