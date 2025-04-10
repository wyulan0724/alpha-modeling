import qlib
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.workflow import R
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.constant import REG_TW
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.report import analysis_model, analysis_position
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from qlib.data.dataset.handler import DataHandlerLP

market = "TW50"
benchmark = "0050"


class MyAlpha158(Alpha158):
    """
    Custom Alpha158 handler with modified label and preprocessing configs.
    """

    def get_label_config(self):
        """Return custom label config that defines next-day returns"""
        return [["Ref($close, -1)/$close - 1"], ["LABEL0"]]

    def __init__(self, **kwargs):
        # Define data preprocessing for training data
        custom_learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSZScoreNorm", "kwargs": {}},
            # {"class": "FilterCol", "kwargs": {"col_list": ["KMID2", "KLOW2", "KSFT2", "RSV5"]}}, # Use these features
        ]

        # Define data preprocessing for prediction data
        custom_infer_processors = [
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "ZScoreNorm", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ]

        # Initialize parent class with custom processors
        super().__init__(
            learn_processors=custom_learn_processors,
            infer_processors=custom_infer_processors,
            **kwargs
        )


XGBModel = {
    "class": "XGBModel",
    "module_path": "qlib.contrib.model.xgboost",
    "kwargs": {
        "eval_metric": "rmse",
        "learning_rate": 0.06424642669823001,
        "max_depth": 5,
        "subsample": 0.8466671252590435,
        "colsample_bytree": 0.962838237653785,
        "reg_alpha": 0.49091134749293946,
        "reg_lambda": 0.4299807160698682
    }
}
LGBModel = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
    }
}

if __name__ == "__main__":
    provider_uri = "~/.qlib/qlib_data/tw_data"  # target_dir
    qlib.init(provider_uri=provider_uri, region=REG_TW)

    task = {
        "model": LGBModel,
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                # Use custom Alpha158 handler
                "handler": MyAlpha158(
                    start_time="2014-12-31",
                    end_time="2024-12-27",
                    fit_start_time="2014-12-31",
                    fit_end_time="2021-12-31",
                    instruments=market
                ),
                # Define train/valid/test splits
                "segments": {
                    "train": ("2014-12-31", "2021-12-31"),
                    "valid": ("2022-01-01", "2022-12-31"),
                    "test": ("2023-01-01", "2024-12-27"),
                },
            },
        }
    }

    # Initialize model and dataset
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    experiment_type = "TRAIN"  # or "PRED", "BACKTEST"
    feature_set = "Alpha158"

    # Start exp
    print(("Starting experiment workflow"))
    with R.start(experiment_name=f"{experiment_type}_{market}_{task['model']['class']}_{feature_set}"):
        # Log model parameters
        print("Logging model parameters")
        R.log_params(**flatten_dict(task))

        # Train model
        print("Training model")
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # Generate predictions
        print("Generating predictions")
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        print("Performing signal analysis")
        sar = SigAnaRecord(recorder)
        sar.generate()

        # Load prediction and label data
        pred_df = recorder.load_object("pred.pkl")
        label_df = recorder.load_object("label.pkl")
        label_df.columns = ["label"]

        # Get normalized label data
        label_df_normalized = dataset.prepare(
            "test", col_set=["label"], data_key=DataHandlerLP.DK_I)
        label_df_normalized.columns = ["label"]

        # Create combined dataframes for raw and normalized data
        pred_label = pd.concat([label_df, pred_df],
                               axis=1, sort=True).reindex(label_df.index)
        pred_label_normalized = pd.concat(
            [label_df_normalized, pred_df], axis=1, sort=True).reindex(label_df.index)

        # Clean up index levels if needed
        for df in [pred_label, pred_label_normalized]:
            if df.index.nlevels > 2:
                df.drop(level=0, inplace=True)

        # Evaluate
        print("Calculating evaluation loss for raw label")
        print(f"raw label: {pred_label['label'].head()}")
        print(f"pred: {pred_label['score'].head()}")
        mse = mean_squared_error(pred_label['label'], pred_label['score'])
        print(f"Mean Squared Error: {mse:.6f}")

        print("Calculating evaluation loss for normalized label")
        print(f"normalized label: {pred_label_normalized['label'].head()}")
        print(f"pred: {pred_label_normalized['score'].head()}")
        mse_normalized = mean_squared_error(
            pred_label_normalized['label'],
            pred_label_normalized['score']
        )
        print(f"Mean Squared Error (Normalized): {mse_normalized:.6f}")

        # Visualize model performance
        output_dir = f"model_performance/{experiment_type}_{market}_{task['model']['class']}_{feature_set}"
        os.makedirs(output_dir, exist_ok=True)

        figures = analysis_model.model_performance_graph(
            pred_label, show_notebook=False)
        for i, fig in enumerate(figures):
            fig.write_html(os.path.join(
                output_dir, f"model_performance_{i}.html"))
            # fig.write_image(os.path.join(output_dir, f"model_performance_{i}.png"))
