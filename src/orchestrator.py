from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.prebuilt import create_react_agent

from src.agents import REGISTERED_TOOLS
from src.memory import memory_manager


CLINICAL_DISCLAIMER = (
    "Disclaimer: MediBot is an AI assistant, not a licensed medical professional. "
    "Always consult a doctor for urgent health concerns."
)

NON_MEDICAL_FALLBACK = (
    "I can only help with medical symptom-checking questions, condition explanations, "
    "severity triage, and dataset-backed precautions. I cannot help with that request."
)


MEDIBOT_REACT_SYSTEM_PROMPT = """You are MediBot, a safe, analytical medical assistant.

You must use the available tools as your only medical knowledge source. Never invent
diseases, precautions, severity scores, risk factors, or diagnoses beyond what tool
observations support.

Tool-selection policy:
- If the user describes symptoms or asks what symptoms might mean, call
  disease_diagnosis_agent.
- If the user asks how serious symptoms are, what they should do, whether it is urgent,
  or mentions red flags such as chest pain or breathing difficulty, call
  symptom_severity_agent.
- If the user asks for an explanation of a condition but did not state a disease name,
  first call disease_diagnosis_agent, then call disease_description_agent with the most
  likely disease from the diagnosis observation.
- If the user asks for precautions, treatment-like self-care, or prevention but did not
  state a disease name, first call disease_diagnosis_agent, then call
  precaution_advisor_agent with the most likely disease from the diagnosis observation.
- For multi-faceted questions, call every required tool before the final answer.

Safety policy:
- Present outputs as educational decision support, not a definitive diagnosis.
- Encourage emergency care when severity observations indicate Emergency or when severe
  red flags are present.
- Keep final answers concise, grounded in observations, and transparent about uncertainty.
- If a user asks about a non-medical or out-of-scope topic, do not call tools. Decline
  briefly and ask for a health-related symptom or condition question.
- Every final answer must include this exact disclaimer: Disclaimer: MediBot is an AI
  assistant, not a licensed medical professional. Always consult a doctor for urgent
  health concerns.
"""


MEDICAL_KEYWORDS = {
    "ache",
    "allergy",
    "blood",
    "breath",
    "breathing",
    "chest",
    "cold",
    "condition",
    "cough",
    "diagnosis",
    "disease",
    "dizzy",
    "doctor",
    "emergency",
    "fever",
    "headache",
    "head",
    "health",
    "heart",
    "hospital",
    "infection",
    "medical",
    "migraine",
    "nausea",
    "pain",
    "precaution",
    "rash",
    "pee",
    "urine",
    "serious",
    "severity",
    "sick",
    "symptom",
    "treatment",
    "urgent",
    "vomiting",
    "nauseous",
    "hurts",
}

NON_MEDICAL_KEYWORDS = {
    "car",
    "engine",
    "homework",
    "investment",
    "movie",
    "programming",
    "recipe",
    "stock",
    "tax",
    "vacation",
    "weather",
}


def _latest_human_content(messages: Iterable[BaseMessage]) -> str:
    for message in reversed(list(messages)):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _tool_messages(messages: Iterable[BaseMessage]) -> list[ToolMessage]:
    return [message for message in messages if isinstance(message, ToolMessage)]


def _called_tool_names(messages: Iterable[BaseMessage]) -> list[str]:
    return [message.name or "" for message in _tool_messages(messages)]


def _parse_tool_json(message: ToolMessage) -> dict[str, Any]:
    try:
        return json.loads(str(message.content))
    except json.JSONDecodeError:
        return {}


def _top_disease_from_observations(messages: Iterable[BaseMessage]) -> str | None:
    for message in reversed(_tool_messages(messages)):
        if message.name == "disease_diagnosis_agent":
            payload = _parse_tool_json(message)
            conditions = payload.get("probable_conditions", [])
            if conditions:
                return conditions[0].get("disease")
    return None


def _disease_from_prior_context(messages: Iterable[BaseMessage]) -> str | None:
    for message in reversed(list(messages)):
        content = str(message.content)
        previous_match = re.search(
            r"Previous likely disease:\s*([^.]+)", content, flags=re.IGNORECASE
        )
        if previous_match:
            return previous_match.group(1).strip()
        closest_match = re.search(
            r"closest matching condition from the local knowledge base is\s+([^.]+)",
            content,
            flags=re.IGNORECASE,
        )
        if closest_match:
            return closest_match.group(1).strip()
    return None


def _has_explicit_disease_query(text: str) -> str | None:
    disease_pattern = re.compile(
        r"\b(migraine|common cold|urinary tract infection|heart attack|pneumonia|"
        r"malaria|allergy|gerd|dengue|typhoid|diabetes|hypertension)\b",
        re.IGNORECASE,
    )
    match = disease_pattern.search(text)
    return match.group(1) if match else None


class DeterministicMedicalReActModel(BaseChatModel):
    """Local tool-calling chat model used for deterministic ReAct verification."""

    @property
    def _llm_type(self) -> str:
        return "deterministic-medibot-react"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DeterministicMedicalReActModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_text = _latest_human_content(messages)
        called_tools = _called_tool_names(messages)
        visible_user_text = user_text.split("Previous likely disease:")[0]
        lower = user_text.lower()

        needs_description = any(
            phrase in lower
            for phrase in ["what is", "explain", "description", "tell me about", "mean"]
        )
        needs_precautions = any(
            phrase in lower
            for phrase in ["what should i do", "precaution", "self care", "prevent", "advice"]
        )
        needs_severity = any(
            phrase in lower
            for phrase in [
                "serious",
                "urgent",
                "emergency",
                "what should i do",
                "high fever",
                "medication dosage",
                "dosage",
                "chest pain",
                "shortness of breath",
                "sharp pain",
            ]
        )
        has_symptom_language = any(
            phrase in lower
            for phrase in [
                "i have",
                "symptom",
                "pain",
                "fever",
                "headache",
                "nausea",
                "cough",
                "breath",
                "dizzy",
            ]
        )
        explicit_disease = _has_explicit_disease_query(visible_user_text)
        disease_for_followup = (
            explicit_disease
            or _top_disease_from_observations(messages)
            or _disease_from_prior_context(messages)
        )

        needs_diagnosis_from_symptoms = has_symptom_language and (
            not explicit_disease or "guarantee" in lower or "diagnose" in lower
        )
        needs_diagnosis_for_dependency = (needs_description or needs_precautions) and not disease_for_followup

        if (
            (needs_diagnosis_from_symptoms or needs_diagnosis_for_dependency)
            and "disease_diagnosis_agent" not in called_tools
        ):
            return self._tool_call(
                "The user gave symptoms or an unstated condition request, so I need to identify likely diseases first.",
                "disease_diagnosis_agent",
                {"symptoms": user_text},
            )

        if needs_severity and "symptom_severity_agent" not in called_tools:
            return self._tool_call(
                "The user asks what to do or has potential red flags, so I need a severity assessment.",
                "symptom_severity_agent",
                {"symptoms": user_text},
            )

        if needs_description and disease_for_followup and "disease_description_agent" not in called_tools:
            return self._tool_call(
                "I have a likely disease and the user needs an explanation, so I should fetch the disease description.",
                "disease_description_agent",
                {"disease_name": disease_for_followup},
            )

        if needs_precautions and disease_for_followup and "precaution_advisor_agent" not in called_tools:
            return self._tool_call(
                "I have a likely disease and the user asks what to do, so I should fetch precautions.",
                "precaution_advisor_agent",
                {"disease_name": disease_for_followup},
            )

        return self._final_answer(messages)

    def _tool_call(self, thought: str, name: str, args: dict[str, Any]) -> ChatResult:
        call_id = f"call_{uuid.uuid4().hex[:10]}"
        message = AIMessage(
            content=thought,
            tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _final_answer(self, messages: list[BaseMessage]) -> ChatResult:
        observations = [_parse_tool_json(message) for message in _tool_messages(messages)]
        disease = _top_disease_from_observations(messages)
        urgency = None
        precautions = None
        description = None

        for message in _tool_messages(messages):
            payload = _parse_tool_json(message)
            if message.name == "symptom_severity_agent":
                urgency = payload.get("urgency_level")
            elif message.name == "precaution_advisor_agent":
                precautions = payload.get("precautions")
            elif message.name == "disease_description_agent":
                description = payload.get("description")

        answer_parts = ["I have the needed tool observations."]
        if disease:
            answer_parts.append(f"The closest matching condition from the local knowledge base is {disease}.")
        if urgency:
            answer_parts.append(f"The triage urgency returned by the severity tool is {urgency}.")
        if description:
            answer_parts.append(f"Description: {description}")
        if precautions:
            answer_parts.append(f"Suggested precautions: {', '.join(precautions)}.")
        if urgency == "Emergency":
            answer_parts.append("Because this was classified as Emergency, seek urgent medical care now.")
        answer_parts.append(
            "This is educational decision support from MediBot's local tools, not a definitive medical diagnosis."
        )
        if not observations:
            answer_parts.append("I do not have enough tool-backed information to answer medically.")

        message = AIMessage(content=" ".join(answer_parts))
        return ChatResult(generations=[ChatGeneration(message=message)])


@dataclass
class ReActRunResult:
    session_id: str
    user_message: str
    context_query: str
    final_answer: str
    messages: list[BaseMessage]
    trace: list[dict[str, Any]]


class MediBotReActOrchestrator:
    """Central ReAct orchestrator that binds MediBot tools and conversation memory."""

    def __init__(self, model: BaseChatModel | None = None, debug: bool = False) -> None:
        self.model = model or DeterministicMedicalReActModel()
        self.tools = REGISTERED_TOOLS
        self.agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=MEDIBOT_REACT_SYSTEM_PROMPT,
            debug=debug,
        )

    def invoke(self, user_message: str, session_id: str = "default") -> ReActRunResult:
        context_query = self._remember_user_turn(user_message, session_id)
        if self._should_decline(user_message, session_id):
            final_answer = with_clinical_disclaimer(NON_MEDICAL_FALLBACK)
            memory_manager.add_agent_turn(final_answer, session_id=session_id)
            return ReActRunResult(
                session_id=session_id,
                user_message=user_message,
                context_query=context_query,
                final_answer=final_answer,
                messages=[],
                trace=[
                    {
                        "type": "fallback",
                        "reason": "Query was ambiguous, non-medical, or outside MediBot's scope.",
                    },
                    {"type": "final", "answer": final_answer},
                ],
            )
        input_messages = self._messages_for_session(session_id, context_query)
        state = self.agent.invoke({"messages": input_messages})
        messages = list(state["messages"])
        current_turn_messages = messages[len(input_messages) :]
        final_answer = with_clinical_disclaimer(str(messages[-1].content) if messages else "")
        if messages:
            messages[-1].content = final_answer
        memory_manager.add_agent_turn(final_answer, session_id=session_id)
        self._update_memory_diagnosis(messages, session_id)
        return ReActRunResult(
            session_id=session_id,
            user_message=user_message,
            context_query=context_query,
            final_answer=final_answer,
            messages=messages,
            trace=format_react_trace(current_turn_messages),
        )

    def stream_trace(self, user_message: str, session_id: str = "default") -> ReActRunResult:
        # LangGraph emits intermediate states while the same ReAct graph runs.
        return self.invoke(user_message=user_message, session_id=session_id)

    def _remember_user_turn(self, user_message: str, session_id: str) -> str:
        state = memory_manager.add_user_turn(user_message, session_id=session_id)
        context_parts = []
        if state.symptom_context:
            context_parts.append(f"Ongoing symptom context: {state.symptom_context}")
        if state.last_diagnosis:
            context_parts.append(f"Previous likely disease: {state.last_diagnosis[0]}")
        if context_parts:
            return f"{user_message}. {'; '.join(context_parts)}"
        return user_message

    def _messages_for_session(self, session_id: str, context_query: str) -> list[BaseMessage]:
        state = memory_manager.get_state(session_id)
        history = list(state.history.messages)
        if history and isinstance(history[-1], HumanMessage):
            history[-1] = HumanMessage(content=context_query)
        elif not history:
            history.append(HumanMessage(content=context_query))
        return history

    def _should_decline(self, user_message: str, session_id: str) -> bool:
        lower = user_message.lower()
        state = memory_manager.get_state(session_id)
        ambiguous_or_unsupported = [
            "cannot describe symptoms",
            "can't describe symptoms",
            "cosmic",
            "exact condition",
            "rare disease",
        ]
        if any(phrase in lower for phrase in ambiguous_or_unsupported):
            return True
        has_context = bool(state.symptoms or state.last_diagnosis)
        if has_context and any(
            phrase in lower
            for phrase in ["what should i do", "explain", "precaution", "serious", "urgent", "also"]
        ):
            return False
        if memory_manager.extract_symptoms(user_message):
            return False
        if any(keyword in lower for keyword in MEDICAL_KEYWORDS):
            return False
        if any(keyword in lower for keyword in NON_MEDICAL_KEYWORDS):
            return True
        return True

    def _update_memory_diagnosis(self, messages: list[BaseMessage], session_id: str) -> None:
        disease = _top_disease_from_observations(messages)
        if disease:
            memory_manager.update_last_diagnosis([disease], session_id=session_id)


def format_react_trace(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                trace.append(
                    {
                        "type": "thought_action",
                        "thought": str(message.content),
                        "action": tool_call["name"],
                        "action_input": tool_call["args"],
                    }
                )
        elif isinstance(message, ToolMessage):
            trace.append(
                {
                    "type": "observation",
                    "tool": message.name,
                    "observation": str(message.content),
                }
            )
        elif isinstance(message, AIMessage):
            trace.append({"type": "final", "answer": str(message.content)})
    return trace


def with_clinical_disclaimer(answer: str) -> str:
    answer = answer.strip()
    if CLINICAL_DISCLAIMER in answer:
        return answer
    if not answer:
        return CLINICAL_DISCLAIMER
    return f"{answer}\n\n{CLINICAL_DISCLAIMER}"
