from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cache import LexicalResponseCache
from src.memory import memory_manager
from src.orchestrator import DeterministicMedicalReActModel, MediBotReActOrchestrator


@pytest.fixture
def mock_llm() -> DeterministicMedicalReActModel:
    return DeterministicMedicalReActModel()


@pytest.fixture
def orchestrator(mock_llm: DeterministicMedicalReActModel) -> MediBotReActOrchestrator:
    return MediBotReActOrchestrator(model=mock_llm)


@pytest.fixture
def isolated_session() -> str:
    session_id = "pytest-session"
    memory_manager.reset(session_id)
    yield session_id
    memory_manager.reset(session_id)


@pytest.fixture
def lexical_cache(tmp_path: Path) -> LexicalResponseCache:
    cache = LexicalResponseCache(db_path=tmp_path / "lexical_cache.sqlite3")
    cache.clear()
    return cache
