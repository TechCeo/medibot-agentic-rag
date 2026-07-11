from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import CachedMediBotOrchestrator, LexicalResponseCache
from src.memory import memory_manager


def timed_call(bot: CachedMediBotOrchestrator, query: str, session_id: str) -> tuple[float, bool, float]:
    start = time.perf_counter()
    result = bot.invoke(query, session_id=session_id)
    latency_ms = (time.perf_counter() - start) * 1000
    return round(latency_ms, 3), result.cache_hit, result.lexical_similarity


def main() -> None:
    cache = LexicalResponseCache()
    cache.clear()
    bot = CachedMediBotOrchestrator(cache=cache)

    miss_query = "I have a throbbing headache and light sensitivity. What could this be?"
    identical_query = "I have a throbbing headache and light sensitivity. What could this be?"
    similar_query = "I have throbbing headache and light sensitivity. What could this be?"

    memory_manager.reset("cache-miss")
    miss_ms, miss_hit, miss_similarity = timed_call(bot, miss_query, "cache-miss")

    memory_manager.reset("cache-hit-identical")
    hit_ms, hit, hit_similarity = timed_call(bot, identical_query, "cache-hit-identical")

    memory_manager.reset("cache-hit-similar")
    similar_ms, similar_hit, similar_similarity = timed_call(bot, similar_query, "cache-hit-similar")

    print("Lexical cache latency verification")
    print(f"Miss:      {miss_ms} ms | cache_hit={miss_hit} | lexical_similarity={miss_similarity}")
    print(f"Identical: {hit_ms} ms | cache_hit={hit} | lexical_similarity={hit_similarity}")
    print(f"Similar:   {similar_ms} ms | cache_hit={similar_hit} | lexical_similarity={similar_similarity}")

    if not hit:
        raise AssertionError("Expected identical query to hit lexical cache.")
    if not similar_hit:
        raise AssertionError("Expected highly similar query to hit lexical cache.")
    if hit_ms >= miss_ms:
        raise AssertionError("Expected cache hit latency to be lower than miss latency.")


if __name__ == "__main__":
    main()
