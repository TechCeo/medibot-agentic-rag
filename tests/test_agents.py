from __future__ import annotations

from src.agents import (
    REGISTERED_TOOLS,
    advise_precautions,
    assess_symptom_severity,
    describe_disease,
    diagnose_disease,
)


def test_plain_agent_functions_return_grounded_results(private_artifacts_required) -> None:
    diagnosis = diagnose_disease("throbbing headache and light sensitivity", top_k=3)
    assert diagnosis["probable_conditions"][0]["disease"] == "Migraine"

    severity = assess_symptom_severity("chest pain")
    assert severity["urgency_level"] == "Emergency"

    description = describe_disease("Migraine")
    assert description["found"] is True
    assert description["matched_disease"] == "Migraine"

    precautions = advise_precautions("Migraine")
    assert precautions["found"] is True
    assert precautions["precautions"]


def test_langchain_tools_are_registered() -> None:
    tool_names = {tool.name for tool in REGISTERED_TOOLS}
    assert tool_names == {
        "disease_diagnosis_agent",
        "symptom_severity_agent",
        "disease_description_agent",
        "precaution_advisor_agent",
    }
