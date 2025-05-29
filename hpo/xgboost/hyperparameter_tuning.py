from qlib.data.dataset.loader import StaticDataLoader
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.eva.alpha import calc_ic
from qlib.utils import init_instance_by_config
import qlib
import optuna
import copy
import pandas as pd
import numpy as np

tw_data_folder_path = "data/tw_data"
parquet_file_path = "data/precomputed_features/tw50_alpha101_label_1d.parquet"
storage_url = "sqlite:///hpo_exp.db"
mlflow_storage_url = "sqlite:///mlflow_exp.db"

DEFAULT_TRAIN_START = "2014-12-31"
DEFAULT_TRAIN_END = "2022-12-31"
DEFAULT_VALID_START = "2023-01-01"
DEFAULT_VALID_END = "2023-12-31"
DEFAULT_TEST_START = "2024-01-01"
DEFAULT_TEST_END = "2024-12-27"

DEFAULT_LEARN_PROCESSORS = [{"class": "DropnaLabel"}]
DEFAULT_INFER_PROCESSORS = [
    {
        "class": "ZScoreNorm",
        "kwargs": {
            "fields_group": "feature",
            "fit_start_time": DEFAULT_TRAIN_START,
            "fit_end_time": DEFAULT_TRAIN_END,
        },
    },
    {"class": "Fillna", "kwargs": {}},
]


# --- Qlib Initialization (MUST RUN FIRST) ---
QLIB_PROVIDER_URI = tw_data_folder_path
QLIB_REGION = "tw"

try:
    print(
        f"Initializing Qlib with provider URI: {QLIB_PROVIDER_URI} and region: {QLIB_REGION}"
    )
    qlib.init(
        provider_uri=QLIB_PROVIDER_URI,
        region=QLIB_REGION,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": mlflow_storage_url, "default_exp_name": "Experiment"},
        },
    )
    print("Qlib initialized successfully.")
    qlib_initialized = True
except NameError:
    print("Error: Qlib is likely not installed.")
    qlib_initialized = False
except Exception as e:
    print(f"Error initializing Qlib: {e}")
    qlib_initialized = False


def _prepare_qlib_dataset(
    data_path: str,
    learn_processors: list,
    infer_processors: list,
    train_start_time: str,
    train_end_time: str,
    valid_start_time: str,
    valid_end_time: str,
    test_start_time: str,
    test_end_time: str,
):
    # 1. Initialize StaticDataLoader
    static_loader = StaticDataLoader(config=data_path)

    # 2. Update infer_processors for ZScoreNorm
    updated_infer_processors = []
    for proc_config in infer_processors:
        proc_copy = copy.deepcopy(proc_config)
        if proc_copy.get("class") == "ZScoreNorm":
            proc_copy["kwargs"] = proc_copy.get("kwargs", {}).copy()
            proc_copy["kwargs"]["fit_start_time"] = train_start_time
            proc_copy["kwargs"]["fit_end_time"] = train_end_time
        updated_infer_processors.append(proc_copy)

    # 3. Construct handler_kwargs
    handler_kwargs = {
        "start_time": train_start_time,
        "end_time": test_end_time,
        "data_loader": static_loader,
        "learn_processors": learn_processors,
        "infer_processors": updated_infer_processors,
    }

    # 4. Construct dataset_config
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": handler_kwargs,
            },
            "segments": {
                "train": (train_start_time, train_end_time),
                "valid": (valid_start_time, valid_end_time),
                "test": (test_start_time, test_end_time),
            },
        },
    }

    # 5. Initialize and return dataset
    dataset = init_instance_by_config(dataset_config)
    return dataset


def _extract_series_from_output(data, col_name="score", default_col_idx=0):
    """
    Extracts a pandas Series from raw model output or dataset label output.
    Handles cases where output is a DataFrame or Series.
    Sorts the series by index.
    """
    if isinstance(data, pd.DataFrame):
        if col_name in data.columns:
            series = data[col_name].copy()
        elif not data.empty and len(data.columns) > default_col_idx:
            series = data.iloc[:, default_col_idx].copy()
        else:
            print(
                f"Warning: Output DataFrame is empty or column '{col_name}' (and default index {default_col_idx}) not found."
            )
            return pd.Series(dtype=np.float64)  # Return empty series
    elif isinstance(data, pd.Series):
        series = data.copy()
    else:
        print(f"Warning: Unsupported data type for extraction: {type(data)}")
        return pd.Series(dtype=np.float64)  # Return empty series
    return series.sort_index()


def _calculate_ic_metrics(pred_series: pd.Series, label_series: pd.Series):
    """
    Aligns prediction and label series, then calculates IC and Rank IC with their standard deviations.

    Args:
        pred_series (pd.Series): Pandas Series of predictions, sorted by index.
        label_series (pd.Series): Pandas Series of labels, sorted by index.

    Returns:
        tuple: (ic_mean, ic_std, ric_mean, ric_std)
    """
    if pred_series.empty or label_series.empty:
        print("Warning: Predictions or labels series is empty before alignment.")
        return np.nan, np.nan, np.nan, np.nan

    common_index = pred_series.index.intersection(label_series.index)

    if common_index.empty:
        print("Warning: No common indices between predictions and labels.")
        return np.nan, np.nan, np.nan, np.nan

    aligned_preds = pred_series.loc[common_index]
    aligned_labels = label_series.loc[common_index]

    if aligned_preds.empty or aligned_labels.empty:
        print("Warning: Empty data after aligning predictions and labels.")
        return np.nan, np.nan, np.nan, np.nan

    # Assuming calc_ic returns (ic_series, ric_series)
    # and handles cases where it might return None if inputs are problematic
    ic_s, ric_s = calc_ic(aligned_preds, aligned_labels)

    ic_mean = ic_s.mean() if ic_s is not None and not np.isnan(ic_s).all() else np.nan
    ic_std = ic_s.std() if ic_s is not None and not np.isnan(ic_s).all() else np.nan
    ric_mean = (
        ric_s.mean() if ric_s is not None and not np.isnan(ric_s).all() else np.nan
    )
    ric_std = ric_s.std() if ric_s is not None and not np.isnan(ric_s).all() else np.nan

    return ic_mean, ic_std, ric_mean, ric_std


# Prepare your dataset once globally
dataset = _prepare_qlib_dataset(
    data_path=parquet_file_path,
    learn_processors=DEFAULT_LEARN_PROCESSORS,
    infer_processors=DEFAULT_INFER_PROCESSORS,
    train_start_time=DEFAULT_TRAIN_START,
    train_end_time=DEFAULT_TRAIN_END,
    valid_start_time=DEFAULT_VALID_START,
    valid_end_time=DEFAULT_VALID_END,
    test_start_time=DEFAULT_TEST_START,
    test_end_time=DEFAULT_TEST_END,
)


def objective(trial, base_params=dict(), use_walk_forward=True):
    current_params = base_params.copy()

    current_params = {
        "eta": trial.suggest_float("eta", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 10.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-8, 0.5, log=True),  # min_split_loss
        "lambda": trial.suggest_float(
            "lambda", 1e-8, 5.0, log=True
        ),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-8, 5.0, log=True),  # L1 regularization
    }

    # 使用滾動窗口驗證評估模型
    avg_ic = (
        fit_and_evaluate_walk_forward(current_params)
        if use_walk_forward
        else fit_and_evaluate(current_params)
    )

    if np.isnan(avg_ic):
        # if IC is nan (Constant Predictions), return a bad value for Optuna (since we are maximizing IC)
        print(f"Trial {trial.number} resulted in NaN IC. Returning -1.0.")
        return -1.0

    return avg_ic


def fit_and_evaluate(model_params):
    # Prepare validation labels once
    raw_labels_valid = dataset.prepare(
        "valid", col_set="label", data_key=DataHandlerLP.DK_L
    )
    label_series_valid = _extract_series_from_output(
        raw_labels_valid, col_name="LABEL0"
    )

    # Create and fit the model
    model_config = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": model_params,
    }
    model = init_instance_by_config(model_config)
    model.fit(dataset)

    # Make predictions
    preds_valid_raw = model.predict(dataset, segment="valid")

    # Calculate IC
    pred_series_valid = _extract_series_from_output(preds_valid_raw, col_name="score")
    ic_mean, ic_std, ric_mean, ric_std = _calculate_ic_metrics(
        pred_series_valid, label_series_valid
    )

    return ic_mean


def final_evaluation(best_params):
    # Initialize the model with the best parameters
    model_config = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": best_params,
    }
    best_model = init_instance_by_config(model_config)

    # Prepare test labels
    raw_labels_test = dataset.prepare(
        "test", col_set="label", data_key=DataHandlerLP.DK_L
    )
    label_series_test = _extract_series_from_output(raw_labels_test, col_name="LABEL0")

    # Fit the model on the training dataset
    best_model.fit(dataset)

    # Make predictions on the test set
    preds_test_raw = best_model.predict(dataset, segment="test")
    pred_series_test = _extract_series_from_output(preds_test_raw, col_name="score")

    ic_mean_test, ic_std_test, ric_mean_test, ric_std_test = _calculate_ic_metrics(
        pred_series_test, label_series_test
    )

    # Calculate ICIR (Information Coefficient Information Ratio)
    ic_ir_test = (
        ic_mean_test / ic_std_test
        if ic_std_test != 0 and not np.isnan(ic_std_test)
        else np.nan
    )
    ric_ir_test = (
        ric_mean_test / ric_std_test
        if ric_std_test != 0 and not np.isnan(ric_std_test)
        else np.nan
    )

    # Display the report
    print(f"Best Parameters: {best_params}")
    print(
        f"Test Set - IC Mean: {ic_mean_test:.6f}, ICIR: {ic_ir_test:.6f}, Rank IC Mean: {ric_mean_test:.6f}, Rank ICIR: {ric_ir_test:.6f}"
    )

    return best_model  # Return the model for feature importance plotting


def plot_feature_importance(model, k=20, figsize=(10, 6), palette="viridis"):
    # Extract and sort importances
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame(importance.items(), columns=["Feature", "Importance"])
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Select top k features
    top_k_df = importance_df.head(k)

    return top_k_df


def hyperparameter_tuning(n_trials=100, study_name=None, use_walk_forward=False):
    best_params = {"eval_metric": "rmse", "random_state": 42}

    validation_method = "walk-forward" if use_walk_forward else "single"
    print(
        f"========== Starting hyperparameter tuning with {validation_method} validation... =========="
    )

    study = optuna.create_study(
        directions=["maximize"],
        storage=storage_url,
        study_name=study_name,
        load_if_exists=True,
    )

    # Create objective function with validation method choice
    def objective_wrapper(trial):
        return objective(trial, best_params, use_walk_forward)

    study.optimize(objective_wrapper, n_trials=n_trials)

    print(f"========== Best parameters: {study.best_trial.params} ==========")
    return study.best_trial


def _create_walk_forward_windows(
    start_year=2015, end_year=2023, train_window_years=5, validation_window_years=1
):
    """
    創建滾動窗口驗證的時間窗口列表

    Args:
        start_year: 開始年份
        end_year: 結束年份
        train_window_years: 訓練窗口年數
        validation_window_years: 驗證窗口年數

    Returns:
        List of tuples: [(train_start, train_end, valid_start, valid_end), ...]
    """
    windows = []

    for valid_start_year in range(start_year + train_window_years, end_year + 1):
        train_start = f"{valid_start_year - train_window_years}-01-01"
        train_end = f"{valid_start_year - 1}-12-31"
        valid_start = f"{valid_start_year}-01-01"
        valid_end = f"{valid_start_year + validation_window_years - 1}-12-31"

        windows.append((train_start, train_end, valid_start, valid_end))

    return windows


def _prepare_walk_forward_dataset(
    data_path: str,
    learn_processors: list,
    infer_processors: list,
    train_start_time: str,
    train_end_time: str,
    valid_start_time: str,
    valid_end_time: str,
):
    """為特定的滾動窗口準備數據集"""
    # 1. Initialize StaticDataLoader
    static_loader = StaticDataLoader(config=data_path)

    # 2. Update infer_processors for ZScoreNorm
    updated_infer_processors = []
    for proc_config in infer_processors:
        proc_copy = copy.deepcopy(proc_config)
        if proc_copy.get("class") == "ZScoreNorm":
            proc_copy["kwargs"] = proc_copy.get("kwargs", {}).copy()
            proc_copy["kwargs"]["fit_start_time"] = train_start_time
            proc_copy["kwargs"]["fit_end_time"] = train_end_time
        updated_infer_processors.append(proc_copy)

    # 3. Construct handler_kwargs
    handler_kwargs = {
        "start_time": train_start_time,
        "end_time": valid_end_time,
        "data_loader": static_loader,
        "learn_processors": learn_processors,
        "infer_processors": updated_infer_processors,
    }

    # 4. Construct dataset_config
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": handler_kwargs,
            },
            "segments": {
                "train": (train_start_time, train_end_time),
                "valid": (valid_start_time, valid_end_time),
            },
        },
    }

    # 5. Initialize and return dataset
    dataset = init_instance_by_config(dataset_config)
    return dataset


def fit_and_evaluate_walk_forward(model_params):
    """
    使用滾動窗口驗證評估模型參數

    Returns:
        float: 所有驗證窗口上IC的平均值
    """
    # 創建滾動窗口
    windows = _create_walk_forward_windows(
        start_year=2015, end_year=2023, train_window_years=5, validation_window_years=1
    )

    ic_scores = []

    for i, (train_start, train_end, valid_start, valid_end) in enumerate(windows):
        print(
            f"Processing window {i+1}/{len(windows)}: Train({train_start} to {train_end}), Valid({valid_start} to {valid_end})"
        )

        try:
            # 為當前窗口準備數據集
            window_dataset = _prepare_walk_forward_dataset(
                data_path=parquet_file_path,
                learn_processors=DEFAULT_LEARN_PROCESSORS,
                infer_processors=DEFAULT_INFER_PROCESSORS,
                train_start_time=train_start,
                train_end_time=train_end,
                valid_start_time=valid_start,
                valid_end_time=valid_end,
            )

            # 創建模型
            model_config = {
                "class": "XGBModel",
                "module_path": "qlib.contrib.model.xgboost",
                "kwargs": model_params,
            }
            model = init_instance_by_config(model_config)

            # 準備驗證標籤
            raw_labels_valid = window_dataset.prepare(
                "valid", col_set="label", data_key=DataHandlerLP.DK_L
            )
            label_series_valid = _extract_series_from_output(
                raw_labels_valid, col_name="LABEL0"
            )

            # 訓練模型
            model.fit(window_dataset)

            # 預測
            preds_valid_raw = model.predict(window_dataset, segment="valid")
            pred_series_valid = _extract_series_from_output(
                preds_valid_raw, col_name="score"
            )

            # 計算IC
            ic_mean, ic_std, ric_mean, ric_std = _calculate_ic_metrics(
                pred_series_valid, label_series_valid
            )

            if not np.isnan(ic_mean):
                ic_scores.append(ic_mean)
                print(f"Window {i+1} IC: {ic_mean:.4f}")
            else:
                print(f"Window {i+1} IC: NaN (skipped)")

        except Exception as e:
            print(f"Error in window {i+1}: {e}")
            continue

    if len(ic_scores) == 0:
        print("No valid IC scores obtained from any window.")
        return np.nan

    # 返回所有窗口IC的平均值
    avg_ic = np.mean(ic_scores)
    print(f"Average IC across {len(ic_scores)} windows: {avg_ic:.4f}")

    return avg_ic


def final_evaluation_walk_forward(best_params):
    # Define the training period for the final model
    # This should be all data available before the test set
    final_train_start_time = "2015-01-01"  # Or your WFV start_year
    final_train_end_time = "2023-12-31"  # Day before final_test_start_time
    final_test_start_time = "2024-01-01"
    final_test_end_time = "2024-12-27"

    print(
        f"Preparing final model training dataset from {final_train_start_time} to {final_train_end_time}"
    )

    # Prepare a new dataset instance for final training
    final_model_dataset = _prepare_qlib_dataset(
        data_path=parquet_file_path,
        learn_processors=DEFAULT_LEARN_PROCESSORS,
        infer_processors=DEFAULT_INFER_PROCESSORS,
        train_start_time=final_train_start_time,
        train_end_time=final_train_end_time,
        valid_start_time=final_train_end_time,  # Dummy valid, not used for fitting final model
        valid_end_time=final_train_end_time,  # Dummy valid
        test_start_time=final_test_start_time,
        test_end_time=final_test_end_time,
    )

    model_config = {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": best_params,
    }
    best_model = init_instance_by_config(model_config)

    # Fit the model on the redefined 'train' segment of final_model_dataset
    print(
        f"Fitting final model with best params on data from {final_train_start_time} to {final_train_end_time}..."
    )
    best_model.fit(final_model_dataset)

    print(
        f"Predicting on test set from {final_test_start_time} to {final_test_end_time}..."
    )
    # Prepare test labels
    raw_labels_test = final_model_dataset.prepare(
        "test", col_set="label", data_key=DataHandlerLP.DK_L
    )

    preds_test_raw = best_model.predict(
        final_model_dataset, segment="test"
    )  # Predict on 'test' segment
    pred_series_test = _extract_series_from_output(preds_test_raw, col_name="score")
    label_series_test = _extract_series_from_output(raw_labels_test, col_name="LABEL0")

    ic_mean_test, ic_std_test, ric_mean_test, ric_std_test = _calculate_ic_metrics(
        pred_series_test, label_series_test
    )

    # Calculate ICIR (Information Coefficient Information Ratio)
    ic_ir_test = (
        ic_mean_test / ic_std_test
        if ic_std_test != 0 and not np.isnan(ic_std_test)
        else np.nan
    )
    ric_ir_test = (
        ric_mean_test / ric_std_test
        if ric_std_test != 0 and not np.isnan(ric_std_test)
        else np.nan
    )

    # Display the report
    print(f"Best Parameters: {best_params}")
    print(
        f"Test Set - IC Mean: {ic_mean_test:.6f}, ICIR: {ic_ir_test:.6f}, Rank IC Mean: {ric_mean_test:.6f}, Rank ICIR: {ric_ir_test:.6f}"
    )

    return best_model


if __name__ == "__main__":
    # Run hyperparameter tuning with walk-forward validation
    use_walk_forward = True
    best_trial = hyperparameter_tuning(
        n_trials=1,
        study_name="xgboost_walk_forward_tuning",
        use_walk_forward=use_walk_forward,
    )
    print("========== Best trial ==========")
    print(best_trial)

    best_params = best_trial.params

    # Perform final evaluation and get the best model
    if use_walk_forward:
        best_model = final_evaluation_walk_forward(best_params)
    else:
        best_model = final_evaluation(best_params)

    # Plot feature importance using the model
    top_k_df = plot_feature_importance(best_model)
    print(f"========== Top {len(top_k_df)} features ==========")
    print(top_k_df)
