from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config


SYMPTOM_COLUMNS_PREFIX = "Symptom_"


@dataclass(frozen=True)
class PipelineData:
    disease_symptoms: pd.DataFrame
    severity_scores: pd.DataFrame
    descriptions: pd.DataFrame
    precautions: pd.DataFrame
    disease_profiles: pd.DataFrame
    overlap_matrix: pd.DataFrame


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace(" _", "_").replace("_ ", "_")
    text = re.sub(r"\s+", " ", text)
    return text


def readable_symptom(symptom: str) -> str:
    return normalize_text(symptom).replace("_", " ")


def normalize_disease(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw_data(dataset_dir: Path = config.DATASET_DIR) -> dict[str, pd.DataFrame]:
    files = {
        "mapping": dataset_dir / "dataset.csv",
        "severity": dataset_dir / "Symptom-severity.csv",
        "description": dataset_dir / "symptom_Description.csv",
        "precaution": dataset_dir / "symptom_precaution.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset files: {missing}")
    return {name: pd.read_csv(path) for name, path in files.items()}


def build_disease_symptom_table(mapping: pd.DataFrame) -> pd.DataFrame:
    df = mapping.copy()
    df["Disease"] = df["Disease"].map(normalize_disease)
    symptom_cols = [col for col in df.columns if col.startswith(SYMPTOM_COLUMNS_PREFIX)]
    long_df = df.melt(
        id_vars=["Disease"],
        value_vars=symptom_cols,
        var_name="symptom_position",
        value_name="symptom",
    )
    long_df["symptom"] = long_df["symptom"].map(normalize_text)
    long_df = long_df[long_df["symptom"] != ""].copy()
    long_df["symptom_readable"] = long_df["symptom"].map(readable_symptom)
    return long_df.drop_duplicates(["Disease", "symptom"]).sort_values(["Disease", "symptom"])


def clean_severity(severity: pd.DataFrame) -> pd.DataFrame:
    df = severity.copy()
    df["symptom"] = df["Symptom"].map(normalize_text)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df = df.dropna(subset=["symptom", "weight"])
    df = df[df["symptom"] != ""]
    # Duplicate rows exist for fluid_overload; average keeps execution deterministic.
    df = df.groupby("symptom", as_index=False)["weight"].mean()
    df["symptom_readable"] = df["symptom"].map(readable_symptom)
    return df.sort_values("symptom")


def clean_descriptions(descriptions: pd.DataFrame) -> pd.DataFrame:
    df = descriptions.copy()
    df["Disease"] = df["Disease"].map(normalize_disease)
    df["Description"] = df["Description"].fillna("").map(lambda value: str(value).strip())
    return df.drop_duplicates("Disease")


def clean_precautions(precautions: pd.DataFrame) -> pd.DataFrame:
    df = precautions.copy()
    df["Disease"] = df["Disease"].map(normalize_disease)
    precaution_cols = [col for col in df.columns if col.startswith("Precaution_")]
    for col in precaution_cols:
        df[col] = df[col].fillna("").map(lambda value: str(value).strip().lower())
    df["precautions"] = df[precaution_cols].apply(
        lambda row: [item for item in row.tolist() if item], axis=1
    )
    return df[["Disease", "precautions"]].drop_duplicates("Disease")


def build_disease_profiles(
    disease_symptoms: pd.DataFrame,
    severity_scores: pd.DataFrame,
    descriptions: pd.DataFrame,
    precautions: pd.DataFrame,
) -> pd.DataFrame:
    enriched = disease_symptoms.merge(
        severity_scores[["symptom", "weight"]], on="symptom", how="left"
    )
    grouped = enriched.groupby("Disease").agg(
        symptoms=("symptom", lambda values: sorted(set(values))),
        symptom_names=("symptom_readable", lambda values: sorted(set(values))),
        symptom_count=("symptom", "nunique"),
        avg_severity=("weight", "mean"),
        max_severity=("weight", "max"),
        missing_severity_count=("weight", lambda values: int(values.isna().sum())),
    )
    profiles = grouped.reset_index()
    profiles["avg_severity"] = profiles["avg_severity"].round(3)
    profiles = profiles.merge(descriptions, on="Disease", how="left")
    profiles = profiles.merge(precautions, on="Disease", how="left")
    profiles["Description"] = profiles["Description"].fillna("")
    profiles["precautions"] = profiles["precautions"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    return profiles.sort_values("Disease").reset_index(drop=True)


def calculate_overlap_matrix(disease_profiles: pd.DataFrame) -> pd.DataFrame:
    disease_to_symptoms = {
        row["Disease"]: set(row["symptoms"]) for _, row in disease_profiles.iterrows()
    }
    diseases = list(disease_to_symptoms)
    matrix = pd.DataFrame(np.eye(len(diseases)), index=diseases, columns=diseases)
    for left, right in combinations(diseases, 2):
        left_symptoms = disease_to_symptoms[left]
        right_symptoms = disease_to_symptoms[right]
        union = left_symptoms | right_symptoms
        score = len(left_symptoms & right_symptoms) / len(union) if union else 0.0
        matrix.loc[left, right] = score
        matrix.loc[right, left] = score
    return matrix


def run_pipeline(dataset_dir: Path = config.DATASET_DIR) -> PipelineData:
    raw = load_raw_data(dataset_dir)
    disease_symptoms = build_disease_symptom_table(raw["mapping"])
    severity_scores = clean_severity(raw["severity"])
    descriptions = clean_descriptions(raw["description"])
    precautions = clean_precautions(raw["precaution"])
    disease_profiles = build_disease_profiles(
        disease_symptoms, severity_scores, descriptions, precautions
    )
    overlap_matrix = calculate_overlap_matrix(disease_profiles)
    return PipelineData(
        disease_symptoms=disease_symptoms,
        severity_scores=severity_scores,
        descriptions=descriptions,
        precautions=precautions,
        disease_profiles=disease_profiles,
        overlap_matrix=overlap_matrix,
    )


def generate_analytics(
    pipeline_data: PipelineData,
    analytics_dir: Path = config.ANALYTICS_DIR,
) -> dict[str, str]:
    analytics_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    outputs: dict[str, str] = {}

    symptom_frequency = (
        pipeline_data.disease_symptoms["symptom_readable"].value_counts().head(25)
    )
    plt.figure(figsize=(12, 8))
    sns.barplot(x=symptom_frequency.values, y=symptom_frequency.index, color="#4c78a8")
    plt.title("Top Symptom Frequency Across Diseases")
    plt.xlabel("Number of Diseases")
    plt.ylabel("Symptom")
    plt.tight_layout()
    path = analytics_dir / "symptom_frequency_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    outputs["symptom_frequency_distribution"] = str(path)

    disease_counts = pipeline_data.disease_profiles.sort_values(
        "symptom_count", ascending=False
    )
    plt.figure(figsize=(12, 10))
    sns.barplot(
        data=disease_counts,
        x="symptom_count",
        y="Disease",
        color="#59a14f",
    )
    plt.title("Symptom Count per Disease")
    plt.xlabel("Unique Symptoms")
    plt.ylabel("Disease")
    plt.tight_layout()
    path = analytics_dir / "disease_symptom_counts.png"
    plt.savefig(path, dpi=160)
    plt.close()
    outputs["disease_symptom_counts"] = str(path)

    plt.figure(figsize=(8, 5))
    sns.histplot(pipeline_data.severity_scores["weight"], bins=7, color="#f28e2b")
    plt.title("Symptom Severity Score Distribution")
    plt.xlabel("Severity Weight")
    plt.ylabel("Symptom Count")
    plt.tight_layout()
    path = analytics_dir / "severity_score_distribution.png"
    plt.savefig(path, dpi=160)
    plt.close()
    outputs["severity_score_distribution"] = str(path)

    overlap = pipeline_data.overlap_matrix
    plt.figure(figsize=(15, 12))
    sns.heatmap(overlap, cmap="mako", vmin=0, vmax=1, xticklabels=True, yticklabels=True)
    plt.title("Disease Symptom Overlap Matrix (Jaccard Similarity)")
    plt.tight_layout()
    path = analytics_dir / "disease_symptom_overlap_heatmap.png"
    plt.savefig(path, dpi=160)
    plt.close()
    outputs["disease_symptom_overlap_heatmap"] = str(path)

    enriched = pipeline_data.disease_symptoms.merge(
        pipeline_data.severity_scores[["symptom", "weight"]], on="symptom", how="left"
    )
    freq_severity = (
        enriched.groupby("symptom_readable")
        .agg(disease_frequency=("Disease", "nunique"), severity_weight=("weight", "mean"))
        .dropna()
        .reset_index()
    )
    corr = freq_severity[["disease_frequency", "severity_weight"]].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="vlag", vmin=-1, vmax=1)
    plt.title("Symptom Frequency vs Severity Correlation")
    plt.tight_layout()
    path = analytics_dir / "severity_frequency_correlation.png"
    plt.savefig(path, dpi=160)
    plt.close()
    outputs["severity_frequency_correlation"] = str(path)

    pipeline_data.disease_profiles.to_json(
        analytics_dir / "disease_profiles.json", orient="records", indent=2
    )
    pipeline_data.overlap_matrix.to_csv(analytics_dir / "disease_overlap_matrix.csv")
    freq_severity.to_csv(analytics_dir / "symptom_frequency_severity.csv", index=False)

    summary = build_statistical_summary(pipeline_data, freq_severity)
    summary_path = analytics_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["eda_summary"] = str(summary_path)
    return outputs


def build_statistical_summary(
    pipeline_data: PipelineData, freq_severity: pd.DataFrame
) -> dict[str, Any]:
    overlap_pairs = []
    overlap = pipeline_data.overlap_matrix.copy()
    for left, right in combinations(overlap.index, 2):
        overlap_pairs.append(
            {
                "disease_a": left,
                "disease_b": right,
                "jaccard_overlap": round(float(overlap.loc[left, right]), 4),
            }
        )
    overlap_pairs = sorted(overlap_pairs, key=lambda item: item["jaccard_overlap"], reverse=True)

    severity_weights = pipeline_data.severity_scores["weight"]
    q1 = float(severity_weights.quantile(0.25))
    q3 = float(severity_weights.quantile(0.75))
    iqr = q3 - q1
    high_outlier_threshold = q3 + 1.5 * iqr
    outliers = pipeline_data.severity_scores[
        pipeline_data.severity_scores["weight"] > high_outlier_threshold
    ]

    return {
        "disease_count": int(pipeline_data.disease_profiles["Disease"].nunique()),
        "unique_symptom_count": int(pipeline_data.disease_symptoms["symptom"].nunique()),
        "disease_symptom_records": int(len(pipeline_data.disease_symptoms)),
        "severity_score_count": int(len(pipeline_data.severity_scores)),
        "missing_severity_matches": int(
            pipeline_data.disease_profiles["missing_severity_count"].sum()
        ),
        "severity_weight_min": float(severity_weights.min()),
        "severity_weight_max": float(severity_weights.max()),
        "severity_weight_mean": round(float(severity_weights.mean()), 3),
        "severity_high_outlier_threshold": round(high_outlier_threshold, 3),
        "severity_outliers": outliers[["symptom", "weight"]].to_dict(orient="records"),
        "frequency_severity_correlation": round(
            float(freq_severity["disease_frequency"].corr(freq_severity["severity_weight"])),
            4,
        ),
        "top_overlapping_disease_pairs": overlap_pairs[:15],
    }


def main() -> None:
    pipeline_data = run_pipeline()
    outputs = generate_analytics(pipeline_data)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
