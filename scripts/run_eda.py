from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import generate_analytics, run_pipeline


def main() -> None:
    pipeline_data = run_pipeline()
    outputs = generate_analytics(pipeline_data)
    print("Generated analytics:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
