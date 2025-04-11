# scripts/run_benchmark.py
import argparse
import datetime
from src.pipelines.fast_benchmark import run_fast_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list", type=str,
                        default="config/benchmark_config.yaml")
    parser.add_argument(
        "--output", type=str, default=f"outputs/benchmark_results/benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    args = parser.parse_args()

    run_fast_benchmark(args.config_list, args.output)
