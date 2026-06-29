from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src import config
from src.data_pipeline import PipelineData, readable_symptom, run_pipeline


QUERY_EXPANSIONS = {
    "light sensitivity": "photophobia visual disturbances blurred and distorted vision headache migraine",
    "sensitive to light": "photophobia visual disturbances blurred and distorted vision headache migraine",
    "throbbing headache": "headache migraine pain behind the eyes visual disturbances",
    "burning urination": "burning micturition bladder discomfort urinary tract infection",
    "painful urination": "burning micturition bladder discomfort urinary tract infection",
    "shortness of breath": "breathlessness chest pain asthma pneumonia heart attack",
    "difficulty breathing": "breathlessness chest pain asthma pneumonia",
    "pain in my chest": "chest pain heart attack breathlessness emergency",
    "sharp pain in my chest": "chest pain heart attack breathlessness emergency",
    "sharp pain in chest": "chest pain heart attack breathlessness emergency",
    "high temperature": "high fever mild fever chills",
    "stomach ache": "stomach pain abdominal pain belly pain",
    "loose stools": "diarrhoea dehydration gastroenteritis",
    "dizzy": "dizziness loss of balance unsteadiness vertigo",
}

SYMPTOM_ALIASES = {
    "visual_disturbances": ["light sensitivity", "photophobia", "vision changes"],
    "blurred_and_distorted_vision": ["blurred vision", "distorted vision"],
    "headache": ["head pain", "throbbing headache"],
    "breathlessness": ["shortness of breath", "difficulty breathing"],
    "chest_pain": ["pain in my chest", "sharp pain in my chest", "sharp pain in chest"],
    "burning_micturition": ["burning urination", "painful urination"],
    "diarrhoea": ["diarrhea", "loose stools"],
    "high_fever": ["fever", "high temperature"],
    "mild_fever": ["low grade fever"],
    "stomach_pain": ["stomach ache"],
    "abdominal_pain": ["belly pain", "abdominal cramps"],
    "spinning_movements": ["vertigo", "room spinning"],
}


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    document_type: str
    disease: str
    text: str
    metadata: dict[str, Any]


def normalize_query(text: str) -> str:
    normalized = text.lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    expansions = []
    for phrase, expansion in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            expansions.append(expansion)
    return f"{normalized} {' '.join(expansions)}".strip()


def symptom_text(symptom: str) -> str:
    base = readable_symptom(symptom)
    aliases = SYMPTOM_ALIASES.get(symptom, [])
    return " ".join([base, *aliases])


def build_documents(pipeline_data: PipelineData) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    severity_lookup = dict(
        zip(pipeline_data.severity_scores["symptom"], pipeline_data.severity_scores["weight"])
    )

    for _, row in pipeline_data.disease_profiles.iterrows():
        symptoms = row["symptoms"]
        symptoms_with_aliases = [symptom_text(symptom) for symptom in symptoms]
        precautions = row["precautions"]
        disease = row["Disease"]
        description = row["Description"]
        severity_terms = [
            f"{symptom_text(symptom)} severity {severity_lookup.get(symptom, 'unknown')}"
            for symptom in symptoms
        ]
        full_text = (
            f"Disease: {disease}. "
            f"Symptoms: {', '.join(symptoms_with_aliases)}. "
            f"Severity context: {'; '.join(severity_terms)}. "
            f"Description: {description}. "
            f"Precautions: {', '.join(precautions)}."
        )
        documents.append(
            {
                "id": f"disease::{disease}",
                "document_type": "disease_profile",
                "disease": disease,
                "text": full_text,
                "metadata": {
                    "symptoms": symptoms,
                    "symptom_count": int(row["symptom_count"]),
                    "avg_severity": float(row["avg_severity"]),
                    "max_severity": float(row["max_severity"]),
                    "precautions": precautions,
                },
            }
        )

        symptom_only_text = (
            f"{disease} symptom pattern includes {', '.join(symptoms_with_aliases)}. "
            f"Likely disease when these symptoms overlap: {disease}."
        )
        documents.append(
            {
                "id": f"symptom_profile::{disease}",
                "document_type": "symptom_profile",
                "disease": disease,
                "text": symptom_only_text,
                "metadata": {"symptoms": symptoms},
            }
        )

        for symptom in symptoms:
            documents.append(
                {
                    "id": f"symptom::{disease}::{symptom}",
                    "document_type": "symptom_record",
                    "disease": disease,
                    "text": (
                        f"Symptom {symptom_text(symptom)} may appear with {disease}. "
                        f"Associated severity score is {severity_lookup.get(symptom, 'unknown')}. "
                        f"Related disease description: {description}"
                    ),
                    "metadata": {
                        "symptom": symptom,
                        "severity": severity_lookup.get(symptom),
                    },
                }
            )
    return documents


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 3),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b",
    )


def build_faiss_index(
    vector_store_dir: Path = config.VECTOR_STORE_DIR,
    pipeline_data: PipelineData | None = None,
) -> dict[str, Any]:
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    pipeline_data = pipeline_data or run_pipeline()
    documents = build_documents(pipeline_data)
    texts = [doc["text"] for doc in documents]

    vectorizer = _make_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    dense = matrix.astype(np.float32).toarray()
    dense = normalize(dense, norm="l2").astype(np.float32)

    index = faiss.IndexFlatIP(dense.shape[1])
    index.add(dense)

    faiss.write_index(index, str(config.INDEX_FILE))
    with config.VECTORIZER_FILE.open("wb") as handle:
        pickle.dump(vectorizer, handle)
    config.METADATA_FILE.write_text(json.dumps(documents, indent=2), encoding="utf-8")

    manifest = {
        "index_type": "faiss.IndexFlatIP",
        "embedding_backend": "sklearn.TfidfVectorizer",
        "similarity": "cosine via L2-normalized inner product",
        "document_count": len(documents),
        "vector_dimension": int(dense.shape[1]),
    }
    config.MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_vector_store(
    vector_store_dir: Path = config.VECTOR_STORE_DIR,
) -> tuple[faiss.Index, TfidfVectorizer, list[dict[str, Any]]]:
    index_path = vector_store_dir / config.INDEX_FILE.name
    vectorizer_path = vector_store_dir / config.VECTORIZER_FILE.name
    metadata_path = vector_store_dir / config.METADATA_FILE.name
    missing = [
        str(path)
        for path in [index_path, vectorizer_path, metadata_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Vector store is incomplete. Missing files: {missing}. Run scripts/build_index.py."
        )
    index = faiss.read_index(str(index_path))
    with vectorizer_path.open("rb") as handle:
        vectorizer = pickle.load(handle)
    documents = json.loads(metadata_path.read_text(encoding="utf-8"))
    return index, vectorizer, documents


def search_medical_records(
    query: str,
    top_k: int = 5,
    vector_store_dir: Path = config.VECTOR_STORE_DIR,
) -> list[SearchResult]:
    index, vectorizer, documents = load_vector_store(vector_store_dir)
    expanded_query = normalize_query(query)
    query_matrix = vectorizer.transform([expanded_query])
    query_dense = query_matrix.astype(np.float32).toarray()
    query_dense = normalize(query_dense, norm="l2").astype(np.float32)
    scores, indices = index.search(query_dense, top_k)

    results: list[SearchResult] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        doc = documents[int(idx)]
        results.append(
            SearchResult(
                rank=rank,
                score=round(float(score), 4),
                document_type=doc["document_type"],
                disease=doc["disease"],
                text=doc["text"],
                metadata=doc["metadata"],
            )
        )
    return results


def search_as_dicts(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return [result.__dict__ for result in search_medical_records(query, top_k)]


def main() -> None:
    manifest = build_faiss_index()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
