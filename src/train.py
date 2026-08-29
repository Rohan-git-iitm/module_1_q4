import argparse
import json
import subprocess
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_NAME = "mnist-mlp-q4"


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("q4-reproducibility")

    df = pd.read_csv("data/mnist.csv")
    y = df["label"].astype(str).values
    X = df.drop(columns=["label"]).values / 255.0
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    with mlflow.start_run(run_name=f"seed{args.seed}") as run:
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.set_tag("git_commit", git_commit())

        model = MLPClassifier(
            hidden_layer_sizes=(args.hidden_size,),
            learning_rate_init=args.lr,
            batch_size=args.batch_size,
            max_iter=1,
            warm_start=True,
            random_state=args.seed,
        )

        for epoch in range(args.epochs):
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, model.predict(X_val))
            mlflow.log_metric("train_loss", model.loss_, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        mlflow.log_metric("final_val_accuracy", val_acc)

        with open("metrics.json", "w") as f:
            json.dump({"final_val_accuracy": val_acc}, f, indent=2)
        mlflow.log_artifact("metrics.json")

        mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        if args.register:
            uri = f"runs:/{run.info.run_id}/model"
            version = mlflow.register_model(uri, MODEL_NAME).version
            MlflowClient().set_registered_model_alias(MODEL_NAME, "staging", version)
            print(f"registered {MODEL_NAME} v{version} with alias 'staging'")

        print(f"run_id={run.info.run_id}  final_val_accuracy={val_acc:.4f}")


if __name__ == "__main__":
    main()
