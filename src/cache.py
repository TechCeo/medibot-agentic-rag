from __future__ import annotations

import json
import pickle
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize

from src import config
from src.orchestrator import MediBotReActOrchestrator, ReActRunResult
from src.vector_service import normalize_query


DEFAULT_CACHE_DB = config.PROJECT_ROOT / "cache" / "lexical_cache.sqlite3"
DEFAULT_LEXICAL_SIMILARITY_THRESHOLD = 0.92


@dataclass(frozen=True)
class LexicalCacheLookupResult:
    hit: bool
    lexical_similarity: float
    query: str | None
    answer: str | None
    trace: list[dict[str, Any]]
    latency_ms: float


@dataclass(frozen=True)
class CachedOrchestratorResult:
    answer: str
    trace: list[dict[str, Any]]
    cache_hit: bool
    lexical_similarity: float
    latency_ms: float
    source_query: str | None


class LexicalResponseCache:
    """SQLite-backed response cache using normalized TF-IDF lexical cosine similarity."""

    def __init__(
        self,
        db_path: Path = DEFAULT_CACHE_DB,
        lexical_vectorizer_path: Path = config.VECTORIZER_FILE,
        lexical_similarity_threshold: float = DEFAULT_LEXICAL_SIMILARITY_THRESHOLD,
    ) -> None:
        self.db_path = db_path
        self.lexical_vectorizer_path = lexical_vectorizer_path
        self.lexical_similarity_threshold = lexical_similarity_threshold
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lexical_vectorizer = self._load_lexical_vectorizer()
        self._init_db()

    def _load_lexical_vectorizer(self) -> Any:
        if not self.lexical_vectorizer_path.exists():
            raise FileNotFoundError(
                f"Missing TF-IDF vectorizer at {self.lexical_vectorizer_path}. "
                "Run scripts/build_index.py first."
            )
        with self.lexical_vectorizer_path.open("rb") as handle:
            return pickle.load(handle)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lexical_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    lexical_vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lexical_cache_created_at "
                "ON lexical_cache(created_at)"
            )
            conn.commit()

    def clear(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM lexical_cache")
            conn.commit()

    def vectorize_query_lexically(self, query: str) -> np.ndarray:
        normalized_query_text = normalize_query(query)
        sparse = self.lexical_vectorizer.transform([normalized_query_text])
        dense = sparse.astype(np.float32).toarray()
        return normalize(dense, norm="l2").astype(np.float32)[0]

    def lookup(self, query: str) -> LexicalCacheLookupResult:
        start = time.perf_counter()
        query_vector = self.vectorize_query_lexically(query)
        best_similarity = -1.0
        best_row: tuple[str, str, str] | None = None

        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT query, answer, trace_json, lexical_vector_json FROM lexical_cache"
            ).fetchall()

        for cached_query, answer, trace_json, lexical_vector_json in rows:
            cached_vector = np.array(json.loads(lexical_vector_json), dtype=np.float32)
            lexical_similarity = float(np.dot(query_vector, cached_vector))
            if lexical_similarity > best_similarity:
                best_similarity = lexical_similarity
                best_row = (cached_query, answer, trace_json)

        latency_ms = (time.perf_counter() - start) * 1000
        if best_row and best_similarity >= self.lexical_similarity_threshold:
            cached_query, answer, trace_json = best_row
            return LexicalCacheLookupResult(
                hit=True,
                lexical_similarity=round(best_similarity, 4),
                query=cached_query,
                answer=answer,
                trace=json.loads(trace_json),
                latency_ms=round(latency_ms, 3),
            )
        return LexicalCacheLookupResult(
            hit=False,
            lexical_similarity=round(max(best_similarity, 0.0), 4),
            query=best_row[0] if best_row else None,
            answer=None,
            trace=[],
            latency_ms=round(latency_ms, 3),
        )

    def store(self, query: str, answer: str, trace: list[dict[str, Any]]) -> None:
        lexical_vector = self.vectorize_query_lexically(query)
        normalized_query_text = normalize_query(query)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO lexical_cache
                    (query, normalized_query, answer, trace_json, lexical_vector_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query,
                    normalized_query_text,
                    answer,
                    json.dumps(trace, ensure_ascii=False),
                    json.dumps(lexical_vector.tolist()),
                    time.time(),
                ),
            )
            conn.commit()


class CachedMediBotOrchestrator:
    """Thin lexical-cache wrapper around MediBotReActOrchestrator."""

    def __init__(
        self,
        orchestrator: MediBotReActOrchestrator | None = None,
        cache: LexicalResponseCache | None = None,
    ) -> None:
        self.orchestrator = orchestrator or MediBotReActOrchestrator()
        self.cache = cache or LexicalResponseCache()

    def invoke(self, user_message: str, session_id: str = "default") -> CachedOrchestratorResult:
        start = time.perf_counter()
        lookup = self.cache.lookup(user_message)
        if lookup.hit and lookup.answer is not None:
            return CachedOrchestratorResult(
                answer=lookup.answer,
                trace=[
                    {
                        "type": "lexical_cache_hit",
                        "lexical_similarity": lookup.lexical_similarity,
                        "source_query": lookup.query,
                    },
                    *lookup.trace,
                ],
                cache_hit=True,
                lexical_similarity=lookup.lexical_similarity,
                latency_ms=round((time.perf_counter() - start) * 1000, 3),
                source_query=lookup.query,
            )

        result: ReActRunResult = self.orchestrator.invoke(user_message, session_id=session_id)
        self.cache.store(user_message, result.final_answer, result.trace)
        return CachedOrchestratorResult(
            answer=result.final_answer,
            trace=result.trace,
            cache_hit=False,
            lexical_similarity=lookup.lexical_similarity,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            source_query=lookup.query,
        )
