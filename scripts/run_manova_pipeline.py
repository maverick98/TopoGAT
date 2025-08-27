#scripts/run_manova_pipeline.py
import argparse
import pandas as pd
from pathlib import Path
import sys
from metrics.manova_runner import run_full_manova_analysis

sys.path.append(str(Path(__file__).resolve().parent.parent))

def main(input_csv, target_col='model', output_dir='results'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    #print(f"\n🔍 Loading dataset: {input_csv}")
    #df = pd.read_csv(input_csv)

    print("\n🧪 Running full MANOVA pipeline...")
    results = run_full_manova_analysis(input_csv, group_col=target_col, output_path=output_path)

    print(f"\n✅ Results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MANOVA pipeline on experiment results.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("--target_col", default="model", help="Column used as the group (default: model)")
    parser.add_argument("--output_dir", default="results", help="Directory to save output (default: results)")

    args = parser.parse_args()
    main(args.input_csv, args.target_col, args.output_dir)
