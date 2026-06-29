from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import generate_analytics, run_pipeline
from src.vector_service import build_faiss_index, search_as_dicts


def main() -> None:
    pipeline_data = run_pipeline()
    generate_analytics(pipeline_data)
    manifest = build_faiss_index(pipeline_data=pipeline_data)
    print("Built vector store:")
    for key, value in manifest.items():
        print(f"- {key}: {value}")

    print("\nSmoke query:")
    for result in search_as_dicts("throbbing headache and light sensitivity", top_k=5):
        print(f"- #{result['rank']} {result['disease']} ({result['score']}) [{result['document_type']}]")


if __name__ == "__main__":
    main()
