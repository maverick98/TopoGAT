import subprocess
from pathlib import Path
from collections import defaultdict

# -------------------------
# Configuration
# -------------------------
datasets = ["MUTAG", "PTC_MR", "ENZYMES", "PROTEINS"]

topogat_variants = ["basic", "node_aware", "gated", "attn", "transformer"]
topogin_variants = ["basic", "node_aware", "gated", "attn", "transformer"]

base_models = {
    "topogat": "gat",
    "topogin": "gin"
}

# -------------------------
# Utilities
# -------------------------

def run_analysis(dataset, topo_model, variant, base_model, runs=10):
    script_path = f"experiments/{topo_model}_vs_{base_model}.py"
    command = [
        "python", script_path,
        "--dataset", dataset,
        "--model", topo_model,
        "--variant", variant,
        "--runs", str(runs)
    ]
    print(f"\n[RUNNING] {' '.join(command)}")
    subprocess.run(command)

def parse_final_verdict(verdict_path: Path):
    if not verdict_path.exists():
        return "❌ No verdict file found.", False

    text = verdict_path.read_text()
    beats_base = "beats the base model" in text.lower()
    return text, beats_base

# -------------------------
# Runner
# -------------------------

def run_full_comparison(topo_model: str, variants, datasets):
    base_model = base_models[topo_model]
    final_summary = defaultdict(list)

    for dataset in datasets:
        print(f"\n========== Dataset: {dataset.upper()} ==========")
        for variant in variants:
            run_analysis(dataset, topo_model, variant, base_model)

            verdict_path = (
                Path("output/results")
                / dataset / topo_model / variant / "final_verdict.txt"
            )
            verdict, beats_base = parse_final_verdict(verdict_path)
            print(f"\n--- Verdict for {variant} ---\n{verdict}\n")

            if beats_base:
                final_summary[dataset].append(variant)

    return final_summary

# -------------------------
# Reporting
# -------------------------

def print_final_summary(topo_model: str, summary: dict):
    base_model = base_models[topo_model]
    print(f"\n=========== OVERALL SUMMARY for {topo_model.upper()} vs {base_model.upper()} ===========\n")

    output_lines = []

    for dataset in datasets:
        winning_variants = summary.get(dataset, [])
        if winning_variants:
            line = f"✅ Dataset {dataset}: {topo_model.upper()} variants that beat base {base_model.upper()} = {', '.join(winning_variants)}"
        else:
            line = f"❌ Dataset {dataset}: No {topo_model.upper()} variants beat base {base_model.upper()}"
        print(line)
        output_lines.append(line)

    # Save to file
    with open(f"final_summary_{topo_model}.txt", "w") as f:
        f.write("\n".join(output_lines))

# -------------------------
# Main Execution
# -------------------------

if __name__ == "__main__":
    # Run TopoGAT
    summary_topogat = run_full_comparison("topogat", topogat_variants, datasets)
    print_final_summary("topogat", summary_topogat)

    # Run TopoGIN
    summary_topogin = run_full_comparison("topogin", topogin_variants, datasets)
    print_final_summary("topogin", summary_topogin)
