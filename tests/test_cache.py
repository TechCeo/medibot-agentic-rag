from __future__ import annotations

from src.cache import CachedMediBotOrchestrator
from src.memory import memory_manager


def test_lexical_cache_hit_bypasses_orchestrator(lexical_cache) -> None:
    bot = CachedMediBotOrchestrator(cache=lexical_cache)
    query = "I have a throbbing headache and light sensitivity. What could this be?"

    memory_manager.reset("cache-test-miss")
    miss = bot.invoke(query, session_id="cache-test-miss")
    assert miss.cache_hit is False

    memory_manager.reset("cache-test-hit")
    hit = bot.invoke(query, session_id="cache-test-hit")
    assert hit.cache_hit is True
    assert hit.lexical_similarity >= 0.92
    assert hit.latency_ms < miss.latency_ms
    assert hit.trace[0]["type"] == "lexical_cache_hit"


def test_lexically_close_query_hits_cache(lexical_cache) -> None:
    bot = CachedMediBotOrchestrator(cache=lexical_cache)
    original = "I have a throbbing headache and light sensitivity. What could this be?"
    close = "I have throbbing headache and light sensitivity. What could this be?"

    bot.invoke(original, session_id="close-cache-source")
    hit = bot.invoke(close, session_id="close-cache-hit")

    assert hit.cache_hit is True
    assert hit.lexical_similarity >= 0.92
