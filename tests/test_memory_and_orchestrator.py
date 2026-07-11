from __future__ import annotations

from src.memory import memory_manager
from src.orchestrator import CLINICAL_DISCLAIMER, NON_MEDICAL_FALLBACK


def test_multi_turn_memory_enriches_follow_up(orchestrator, isolated_session: str) -> None:
    first = orchestrator.invoke(
        "I have a throbbing headache and light sensitivity.",
        session_id=isolated_session,
    )
    assert "Migraine" in first.final_answer

    second = orchestrator.invoke(
        "I also have a fever and nausea.",
        session_id=isolated_session,
    )
    snapshot = memory_manager.snapshot(isolated_session)
    assert "high_fever" in snapshot["symptoms"]
    assert "nausea" in snapshot["symptoms"]
    assert "disease_diagnosis_agent" in [step.get("action") for step in second.trace]


def test_red_flag_query_routes_to_diagnosis_and_severity(orchestrator, isolated_session: str) -> None:
    result = orchestrator.invoke(
        "I have a sharp pain in my chest, what should I do?",
        session_id=isolated_session,
    )
    actions = [step.get("action") for step in result.trace if step["type"] == "thought_action"]
    assert "disease_diagnosis_agent" in actions
    assert "symptom_severity_agent" in actions
    assert "Emergency" in result.final_answer
    assert CLINICAL_DISCLAIMER in result.final_answer


def test_non_medical_query_uses_fallback_without_tools(orchestrator, isolated_session: str) -> None:
    result = orchestrator.invoke("Tell me how to hotwire a car.", session_id=isolated_session)
    assert result.trace[0]["type"] == "fallback"
    assert NON_MEDICAL_FALLBACK in result.final_answer
    assert CLINICAL_DISCLAIMER in result.final_answer
