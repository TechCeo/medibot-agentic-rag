from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.memory import memory_manager
from src.orchestrator import MediBotReActOrchestrator


DEFAULT_HOST = os.getenv("MEDIBOT_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("MEDIBOT_PORT", os.getenv("PORT", "7860")))
DEFAULT_SHARE = os.getenv("MEDIBOT_SHARE", "false").lower() in {"1", "true", "yes"}
MODEL_MODE = os.getenv("MEDIBOT_MODEL_MODE", "deterministic-local")

ORCHESTRATOR = MediBotReActOrchestrator()


def _new_session_id() -> str:
    return f"ui-{uuid.uuid4().hex}"


def _summarize_observation(raw_observation: str) -> str:
    try:
        payload = json.loads(raw_observation)
    except json.JSONDecodeError:
        return raw_observation[:900]

    if "probable_conditions" in payload:
        conditions = payload.get("probable_conditions", [])[:3]
        ranked = [
            f"{item.get('disease')} ({item.get('confidence')}, score {item.get('best_score')})"
            for item in conditions
        ]
        return "Probable conditions: " + "; ".join(ranked)
    if "urgency_level" in payload:
        return (
            f"Urgency: {payload.get('urgency_level')}; "
            f"max severity {payload.get('max_severity_weight')}; "
            f"{payload.get('triage_justification')}"
        )
    if "description" in payload:
        return f"{payload.get('matched_disease')}: {payload.get('description')}"
    if "precautions" in payload:
        return (
            f"{payload.get('matched_disease')} precautions: "
            + ", ".join(payload.get("precautions", []))
        )
    return json.dumps(payload, ensure_ascii=False)[:900]


def render_trace(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "No agent steps recorded yet."

    lines: list[str] = []
    step_number = 1
    for step in trace:
        if step["type"] == "thought_action":
            lines.append(f"### Step {step_number}: Thought -> Action")
            lines.append(f"**Thought:** {step['thought']}")
            lines.append(f"**Action:** `{step['action']}`")
            lines.append(f"**Action input:** `{step['action_input']}`")
            step_number += 1
        elif step["type"] == "observation":
            lines.append(f"**Observation from `{step['tool']}`:**")
            lines.append(_summarize_observation(step["observation"]))
        elif step["type"] == "fallback":
            lines.append("### Safety Fallback")
            lines.append(step["reason"])
        elif step["type"] == "final":
            lines.append("### Final Response")
            lines.append(step["answer"])
    return "\n\n".join(lines)


def chat(
    user_message: str,
    chat_history: list[dict[str, str]] | None,
    session_id: str | None,
) -> tuple[str, list[dict[str, str]], str, str]:
    if not session_id:
        session_id = _new_session_id()
        memory_manager.reset(session_id)

    chat_history = chat_history or []
    if not user_message.strip():
        return "", chat_history, "Please enter a symptom or medical question.", session_id

    result = ORCHESTRATOR.invoke(user_message.strip(), session_id=session_id)
    chat_history = [
        *chat_history,
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": result.final_answer},
    ]
    return "", chat_history, render_trace(result.trace), session_id


def clear_session() -> tuple[list[dict[str, str]], str, str]:
    session_id = _new_session_id()
    memory_manager.reset(session_id)
    return [], "No agent steps recorded yet.", session_id


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MediBot") as demo:
        session_state = gr.State(_new_session_id())

        gr.Markdown(
            "# MediBot\n"
            "AI-powered symptom checking with retrieval-backed specialist tools and transparent ReAct routing."
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=560,
                    buttons=["copy"],
                )
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="Describe symptoms or ask a medical question...",
                        label="Message",
                        scale=5,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.Button("New session")
            with gr.Column(scale=2):
                gr.Markdown("## Thought/Action Log")
                trace_log = gr.Markdown("No agent steps recorded yet.")
                gr.Markdown(
                    "Model mode: `"
                    + MODEL_MODE
                    + "`\n\n"
                    "The log shows which specialist tools were triggered during the latest turn."
                )

        send_button.click(
            chat,
            inputs=[textbox, chatbot, session_state],
            outputs=[textbox, chatbot, trace_log, session_state],
        )
        textbox.submit(
            chat,
            inputs=[textbox, chatbot, session_state],
            outputs=[textbox, chatbot, trace_log, session_state],
        )
        clear_button.click(
            clear_session,
            inputs=[],
            outputs=[chatbot, trace_log, session_state],
        )
    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(
        server_name=DEFAULT_HOST,
        server_port=DEFAULT_PORT,
        share=DEFAULT_SHARE,
        theme=gr.themes.Soft(),
    )
