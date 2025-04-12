## Install Qlib and required dependencies:

- [Install Qlib by pip or from source code](https://qlib.readthedocs.io/en/latest/start/installation.html)
- Install packages needed for your chosen models

  ```
  pip install lightgbm xgboost
  ```

## Prepare data:

- The project expects Taiwan stock market data in the `data/tw_data/` directory.
- The data should include calendars, features, and instrument definitions.

## Run Fast Benchmark:

- Prepare the `config/benchmark_config.yaml` file with the desired benchmark settings.
- Use the `scripts/run_benchmark.py` script to execute a fast benchmark.
- Example command:

  ```
  python scripts/run_benchmark.py --config config/benchmark_config.yaml
  ```
