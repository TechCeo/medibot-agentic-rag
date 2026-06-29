from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import memory_manager
from src.orchestrator import CLINICAL_DISCLAIMER, MediBotReActOrchestrator


def main() -> None:
    orchestrator = MediBotReActOrchestrator()

    fallback_session = "stage5-fallback"
    memory_manager.reset(fallback_session)
    fallback = orchestrator.invoke(
        "How do I fix a broken car engine?",
        session_id=fallback_session,
    )
    print("Fallback answer:")
    print(fallback.final_answer)
    print("Fallback trace:", fallback.trace)
    assert fallback.trace[0]["type"] == "fallback"
    assert CLINICAL_DISCLAIMER in fallback.final_answer

    medical_session = "stage5-medical"
    memory_manager.reset(medical_session)
    medical = orchestrator.invoke(
        "I have a sharp pain in my chest, what should I do?",
        session_id=medical_session,
    )
    actions = [step.get("action") for step in medical.trace if step["type"] == "thought_action"]
    print("\nMedical actions:", actions)
    print("Medical answer:")
    print(medical.final_answer)
    assert "disease_diagnosis_agent" in actions
    assert "symptom_severity_agent" in actions
    assert CLINICAL_DISCLAIMER in medical.final_answer


if __name__ == "__main__":
    main()
