from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Medibot_dataset"
ANALYTICS_DIR = PROJECT_ROOT / "analytics"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

DATASET_FILE = DATASET_DIR / "dataset.csv"
SEVERITY_FILE = DATASET_DIR / "Symptom-severity.csv"
DESCRIPTION_FILE = DATASET_DIR / "symptom_Description.csv"
PRECAUTION_FILE = DATASET_DIR / "symptom_precaution.csv"

INDEX_FILE = VECTOR_STORE_DIR / "medibot.faiss"
VECTORIZER_FILE = VECTOR_STORE_DIR / "tfidf_vectorizer.pkl"
METADATA_FILE = VECTOR_STORE_DIR / "documents.json"
MANIFEST_FILE = VECTOR_STORE_DIR / "manifest.json"
