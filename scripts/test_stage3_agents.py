from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import (
    advise_precautions,
    assess_symptom_severity,
    describe_disease,
    diagnose_disease,
    disease_description_agent,
    disease_diagnosis_agent,
    precaution_advisor_agent,
    symptom_severity_agent,
    update_conversation_context,
)
from src.memory import memory_manager


def main() -> None:
    print("Plain function checks")
    diagnosis = diagnose_disease("throbbing headache and light sensitivity", top_k=3)
    print("diagnosis:", [item["disease"] for item in diagnosis["probable_conditions"]])
    print("severity:", assess_symptom_severity("headache and visual disturbances")["urgency_level"])
    print("description:", describe_disease("Migraine")["matched_disease"])
    print("precautions:", advise_precautions("Migraine")["precautions"])

    print("\nLangChain tool registration checks")
    print("tools:", [tool.name for tool in [
        disease_diagnosis_agent,
        symptom_severity_agent,
        disease_description_agent,
        precaution_advisor_agent,
    ]])

    print("\nMemory enrichment check")
    memory_manager.reset("stage3-demo")
    first = update_conversation_context(
        "I have a throbbing headache and light sensitivity.",
        session_id="stage3-demo",
    )
    second = update_conversation_context(
        "I also have a fever and nausea.",
        session_id="stage3-demo",
    )
    print("turn 1 symptoms:", first["memory"]["symptom_context"])
    print("turn 1 top disease:", first["diagnosis"]["probable_conditions"][0]["disease"])
    print("turn 2 symptoms:", second["memory"]["symptom_context"])
    print("turn 2 top disease:", second["diagnosis"]["probable_conditions"][0]["disease"])
    print("turn 2 context query:", second["context_query"])


if __name__ == "__main__":
    main()
