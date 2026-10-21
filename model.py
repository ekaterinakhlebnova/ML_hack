

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split


def compute_metric(y_true, y_pred, metric):
    if metric == "rmse":
        return mean_squared_error(y_true, y_pred) ** 0.5
    if metric == "macro_f1":
        return f1_score(y_true, y_pred, average="macro")
    raise ValueError(f"Неизвестная метрика: {metric}")


def is_better(new_score, old_score, metric):
    if old_score is None:
        return True
    return new_score < old_score if metric == "rmse" else new_score > old_score


def train_and_predict(task, params):
    train = pd.read_csv(task["train_path"])
    test = pd.read_csv(task["test_path"])
    target_col = task["target_col"]
    id_col = task["id_col"]

    y = train[target_col]
    X = train.drop(columns=[id_col, target_col])
    X_test = test.drop(columns=[id_col])


    n_train = len(X)
    both = pd.concat([X, X_test], axis=0)
    for col in both.select_dtypes(exclude="number").columns:
        both[col] = both[col].astype("category")
    X = both.iloc[:n_train]
    X_test = both.iloc[n_train:]

    #я поделила тренировочную выборку на 80% для тренировки и 20% для тестирования
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_target = params.get("log_target", False)
    task_type = task["task_type"]

    if task_type == "regression":
        model = LGBMRegressor(verbose=-1)
        y_fit = np.log1p(y_train) if log_target else y_train
    else:
        model = LGBMClassifier(verbose=-1, class_weight=params.get("class_weight"))
        y_fit = y_train

    model.fit(X_train, y_fit)

    pred_valid = model.predict(X_valid)
    if task_type == "regression" and log_target:
        pred_valid = np.expm1(pred_valid)
    score = compute_metric(y_valid, pred_valid, task["metric"])

    predictions = model.predict(X_test)
    if task_type == "regression" and log_target:
        predictions = np.expm1(predictions)

    return {"score": float(score), "predictions": predictions.tolist()}


if __name__ == "__main__":
    tasks = {
        "regression": {
            "train_path": "data/regression/train.csv",
            "test_path": "data/regression/test.csv",
            "id_col": "id",
            "target_col": "site_eui",
            "task_type": "regression",
            "metric": "rmse",
        },
        "classification": {
            "train_path": "data/multiclass/train.csv",
            "test_path": "data/multiclass/test.csv",
            "id_col": "id",
            "target_col": "attack_cat",
            "task_type": "classification",
            "metric": "macro_f1",
        },
    }
    for name, task in tasks.items():
        result = train_and_predict(task, {})
        print(name, "score:", round(result["score"], 4), "predictions:", len(result["predictions"]), "первые:", result["predictions"][:3])
