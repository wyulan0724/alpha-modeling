import argparse
import datetime
import os
from src.pipelines.fast_benchmark import run_fast_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_list", type=str,
                        default="config/benchmark_config.yaml",
                        help="Path to the benchmark configuration YAML file.")
    default_output_dir = "outputs/benchmark_results"
    default_output_filename = f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    default_output_path = os.path.join(
        default_output_dir, default_output_filename)

    parser.add_argument(
        "--output", type=str,
        default=default_output_path,
        help="Path to save the benchmark report JSON file."
    )
    args = parser.parse_args()

    # Ensure the output directory exists before running the benchmark
    output_dir = os.path.dirname(args.output)
    if output_dir:  # Avoid creating directory if output is in current directory
        os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Fast Benchmark...")
    print(f"Benchmark Config: {args.config_list}")
    print(f"Output Report: {args.output}")

    run_fast_benchmark(args.config_list, args.output)

    print(f"Fast Benchmark finished.")
