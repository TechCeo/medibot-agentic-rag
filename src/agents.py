from __future__ import annotations

import json
from difflib import get_close_matches
from typing import Any

from langchain_core.tools import tool

from src.data_pipeline import PipelineData, run_pipeline
from src.memory import memory_manager
from src.vector_service import search_as_dicts


def _pipeline_data() -> PipelineData:
    return run_pipeline()


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _match_disease_name(disease_name: str, pipeline_data: PipelineData | None = None) -> str | None:
    pipeline_data = pipeline_data or _pipeline_data()
    diseases = list(pipeline_data.disease_profiles["Disease"])
    normalized = {disease.lower(): disease for disease in diseases}
    key = disease_name.strip().lower()
    if key in normalized:
        return normalized[key]
    matches = get_close_matches(key, normalized.keys(), n=1, cutoff=0.55)
    return normalized[matches[0]] if matches else None


def diagnose_disease(symptoms: str, top_k: int = 5) -> dict[str, Any]:
    results = search_as_dicts(symptoms, top_k=max(top_k, 5))
    disease_scores: dict[str, dict[str, Any]] = {}
    for result in results:
        disease = result["disease"]
        current = disease_scores.setdefault(
            disease,
            {
                "disease": disease,
                "best_score": 0.0,
                "evidence": [],
                "matched_symptoms": set(),
            },
        )
        current["best_score"] = max(current["best_score"], float(result["score"]))
        current["evidence"].append(
            {
                "rank": result["rank"],
                "score": result["score"],
                "document_type": result["document_type"],
            }
        )
        metadata = result.get("metadata", {})
        if "symptoms" in metadata:
            current["matched_symptoms"].update(metadata["symptoms"])
        if "symptom" in metadata and metadata["symptom"]:
            current["matched_symptoms"].add(metadata["symptom"])

    ranked = sorted(
        disease_scores.values(), key=lambda item: item["best_score"], reverse=True
    )[:top_k]
    for item in ranked:
        item["matched_symptoms"] = sorted(item["matched_symptoms"])
        item["confidence"] = (
            "High" if item["best_score"] >= 0.35 else "Medium" if item["best_score"] >= 0.2 else "Low"
        )
    return {
        "query": symptoms,
        "probable_conditions": ranked,
        "clinical_note": (
            "This is retrieval-based decision support, not a medical diagnosis. "
            "Escalate urgent or worsening symptoms to a qualified clinician."
        ),
    }


def assess_symptom_severity(symptoms: str, session_id: str = "default") -> dict[str, Any]:
    pipeline_data = _pipeline_data()
    severity_lookup = dict(
        zip(pipeline_data.severity_scores["symptom"], pipeline_data.severity_scores["weight"])
    )
    extracted = memory_manager.extract_symptoms(symptoms)
    if not extracted:
        extracted = memory_manager.get_state(session_id).symptoms

    scored = [
        {
            "symptom": symptom,
            "readable_symptom": symptom.replace("_", " "),
            "severity_weight": float(severity_lookup.get(symptom, 0.0)),
        }
        for symptom in extracted
    ]
    max_score = max([item["severity_weight"] for item in scored], default=0.0)
    avg_score = sum(item["severity_weight"] for item in scored) / len(scored) if scored else 0.0
    emergency_terms = {
        "chest_pain",
        "coma",
        "acute_liver_failure",
        "stomach_bleeding",
        "weakness_in_limbs",
        "weakness_of_one_body_side",
    }
    if any(item["symptom"] in emergency_terms for item in scored) or max_score >= 7:
        urgency = "Emergency"
    elif max_score >= 6 or avg_score >= 5:
        urgency = "High"
    elif max_score >= 4 or avg_score >= 3:
        urgency = "Medium"
    else:
        urgency = "Low"

    return {
        "symptoms": scored,
        "urgency_level": urgency,
        "max_severity_weight": max_score,
        "average_severity_weight": round(avg_score, 2),
        "triage_justification": (
            f"Urgency is {urgency} because the highest matched symptom severity is "
            f"{max_score} and the average matched severity is {avg_score:.2f}."
        ),
        "safety_note": (
            "Seek emergency care for severe chest pain, trouble breathing, fainting, "
            "one-sided weakness, confusion, uncontrolled bleeding, or rapidly worsening symptoms."
        ),
    }


def describe_disease(disease_name: str) -> dict[str, Any]:
    pipeline_data = _pipeline_data()
    matched = _match_disease_name(disease_name, pipeline_data)
    if not matched:
        return {
            "requested_disease": disease_name,
            "found": False,
            "message": "No close disease match was found in the local MediBot knowledge base.",
        }

    row = pipeline_data.disease_profiles[
        pipeline_data.disease_profiles["Disease"] == matched
    ].iloc[0]
    return {
        "requested_disease": disease_name,
        "matched_disease": matched,
        "found": True,
        "description": row["Description"],
        "common_symptoms": row["symptom_names"],
        "risk_factors_or_context": (
            "The current dataset provides disease descriptions and symptom patterns; "
            "formal risk-factor data is not separately modeled yet."
        ),
    }


def advise_precautions(disease_name: str) -> dict[str, Any]:
    pipeline_data = _pipeline_data()
    matched = _match_disease_name(disease_name, pipeline_data)
    if not matched:
        return {
            "requested_disease": disease_name,
            "found": False,
            "message": "No close disease match was found for precaution lookup.",
        }

    row = pipeline_data.disease_profiles[
        pipeline_data.disease_profiles["Disease"] == matched
    ].iloc[0]
    return {
        "requested_disease": disease_name,
        "matched_disease": matched,
        "found": True,
        "precautions": row["precautions"],
        "self_care_note": (
            "These precautions come from the local dataset and should support, not replace, "
            "professional medical advice."
        ),
    }


def update_conversation_context(user_message: str, session_id: str = "default") -> dict[str, Any]:
    context_query = memory_manager.build_context_query(user_message, session_id=session_id)
    diagnosis = diagnose_disease(context_query, top_k=3)
    diseases = [item["disease"] for item in diagnosis["probable_conditions"]]
    memory_manager.update_last_diagnosis(diseases, session_id=session_id)
    return {
        "context_query": context_query,
        "memory": memory_manager.snapshot(session_id),
        "diagnosis": diagnosis,
    }


@tool
def disease_diagnosis_agent(symptoms: str) -> str:
    """Use this tool when the user describes one or more symptoms and needs likely matching medical conditions from MediBot's local FAISS retrieval index. The input should be the full natural-language symptom description, including any symptoms remembered from prior turns if available. The tool returns ranked probable diseases, retrieval confidence, matched symptom evidence, and a safety note. This tool is best for initial differential-condition lookup, symptom-overlap matching, and synonym-tolerant retrieval such as 'light sensitivity' matching migraine-related records."""
    return _json(diagnose_disease(symptoms))


@tool
def symptom_severity_agent(symptoms: str) -> str:
    """Use this tool when the user needs triage guidance or urgency assessment for a symptom set. The input should be a natural-language list or sentence describing current symptoms. The tool maps recognized symptoms to the processed severity-score table and returns an urgency level of Low, Medium, High, or Emergency with a transparent justification. Call this for worsening symptoms, red flags, or whenever the system needs to decide whether self-care advice is enough or urgent medical care should be recommended."""
    return _json(assess_symptom_severity(symptoms))


@tool
def disease_description_agent(disease_name: str) -> str:
    """Use this tool when a probable disease or condition name is known and the user wants a clear patient-friendly explanation. The input should be a single disease name, such as 'Migraine', 'Common Cold', or 'Urinary tract infection'. The tool performs tolerant disease-name matching against MediBot's processed medical knowledge base and returns the local description plus common symptoms. This tool should not be used for free-form symptom diagnosis; use disease_diagnosis_agent first for that."""
    return _json(describe_disease(disease_name))


@tool
def precaution_advisor_agent(disease_name: str) -> str:
    """Use this tool when a likely disease or condition has already been identified and the user needs practical precautionary or self-care steps from the local MediBot knowledge base. The input should be a single condition name. The tool returns dataset-backed precautions for the matched disease and a reminder that these steps do not replace clinician guidance. This tool is intended for after diagnosis retrieval, not for predicting a disease from symptoms."""
    return _json(advise_precautions(disease_name))


REGISTERED_TOOLS = [
    disease_diagnosis_agent,
    symptom_severity_agent,
    disease_description_agent,
    precaution_advisor_agent,
]
