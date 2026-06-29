from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.chat_history import InMemoryChatMessageHistory

from src.data_pipeline import run_pipeline
from src.vector_service import SYMPTOM_ALIASES, normalize_query


def _readable_symptom(symptom: str) -> str:
    return symptom.replace("_", " ").strip().lower()


def _build_symptom_vocabulary() -> dict[str, str]:
    pipeline_data = run_pipeline()
    vocabulary: dict[str, str] = {}
    for symptom in sorted(pipeline_data.severity_scores["symptom"].unique()):
        vocabulary[_readable_symptom(symptom)] = symptom
        vocabulary[symptom.lower()] = symptom
    for canonical, aliases in SYMPTOM_ALIASES.items():
        vocabulary[_readable_symptom(canonical)] = canonical
        for alias in aliases:
            vocabulary[alias.lower()] = canonical
    return vocabulary


SYMPTOM_VOCABULARY = _build_symptom_vocabulary()


@dataclass
class ConversationState:
    session_id: str
    history: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)
    symptoms: list[str] = field(default_factory=list)
    last_diagnosis: list[str] = field(default_factory=list)

    @property
    def symptom_context(self) -> str:
        return ", ".join(_readable_symptom(symptom) for symptom in self.symptoms)


class MediBotMemoryManager:
    """Structured memory store for multi-turn MediBot symptom conversations."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_state(self, session_id: str = "default") -> ConversationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str = "default") -> None:
        self._sessions.pop(session_id, None)

    def extract_symptoms(self, text: str) -> list[str]:
        normalized = normalize_query(text)
        padded = f" {normalized} "
        found: list[str] = []
        for phrase, canonical in SYMPTOM_VOCABULARY.items():
            pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
            if re.search(pattern, padded):
                found.append(canonical)
        return sorted(set(found))

    def add_user_turn(self, user_message: str, session_id: str = "default") -> ConversationState:
        state = self.get_state(session_id)
        state.history.add_user_message(user_message)
        for symptom in self.extract_symptoms(user_message):
            if symptom not in state.symptoms:
                state.symptoms.append(symptom)
        return state

    def add_agent_turn(self, response: str, session_id: str = "default") -> ConversationState:
        state = self.get_state(session_id)
        state.history.add_ai_message(response)
        return state

    def update_last_diagnosis(
        self, diseases: list[str], session_id: str = "default"
    ) -> ConversationState:
        state = self.get_state(session_id)
        state.last_diagnosis = diseases
        return state

    def build_context_query(self, new_message: str, session_id: str = "default") -> str:
        state = self.add_user_turn(new_message, session_id=session_id)
        if state.symptom_context:
            return f"{new_message}. Ongoing symptom context: {state.symptom_context}"
        return new_message

    def snapshot(self, session_id: str = "default") -> dict[str, Any]:
        state = self.get_state(session_id)
        return {
            "session_id": state.session_id,
            "symptoms": state.symptoms,
            "symptom_context": state.symptom_context,
            "last_diagnosis": state.last_diagnosis,
            "message_count": len(state.history.messages),
            "messages": [
                {"type": message.type, "content": message.content}
                for message in state.history.messages
            ],
        }


memory_manager = MediBotMemoryManager()
