# src/pipelines/fast_benchmark.py
import yaml
from src.pipelines.run_workflow import run_workflow
import pandas as pd
import os
import datetime


def run_fast_benchmark(config_list_path: str, output_path: str = f"outputs/benchmark_results/benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"):
    """Run multiple Qlib workflows defined in a YAML list and summarize results."""
    with open(config_list_path, "r") as f:
        config_list = yaml.safe_load(f)["configs"]

    results = []
    for config_path in config_list:
        print(f"Running: {config_path}")
        try:
            result = run_workflow(config_path)
            results.append(result)
        except Exception as e:
            print(f"Failed on {config_path}: {e}")

    # Convert results to DataFrame, then to JSON
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JSON instead of Excel
    df.to_json(output_path, orient="records", indent=4)
    print(f"[Done] Benchmark report saved to {output_path}")
