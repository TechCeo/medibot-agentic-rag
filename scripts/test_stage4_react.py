from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import memory_manager
from src.orchestrator import MediBotReActOrchestrator


def print_trace(label: str, result) -> None:
    print(f"\n=== {label} ===")
    print("Context query:")
    print(result.context_query)
    print("\nThought -> Action -> Observation chain:")
    for step in result.trace:
        if step["type"] == "thought_action":
            print(f"Thought: {step['thought']}")
            print(f"Action: {step['action']}")
            print(f"Action Input: {step['action_input']}")
        elif step["type"] == "observation":
            observation = step["observation"].replace("\n", " ")
            print(f"Observation ({step['tool']}): {observation[:700]}")
        elif step["type"] == "final":
            print(f"Final: {step['answer']}")


def main() -> None:
    session_id = "stage4-demo"
    memory_manager.reset(session_id)
    orchestrator = MediBotReActOrchestrator()

    first = orchestrator.invoke(
        "I have a sharp pain in my chest, what should I do?",
        session_id=session_id,
    )
    print_trace("Turn 1: chest pain urgency", first)

    second = orchestrator.invoke(
        "Can you explain what this might be and precautions I should take?",
        session_id=session_id,
    )
    print_trace("Turn 2: dependent description and precautions", second)

    print("\nMemory snapshot:")
    print(memory_manager.snapshot(session_id))


if __name__ == "__main__":
    main()
