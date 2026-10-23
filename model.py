import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split


METRICS = {
    "rmse": lambda t, p: skm.mean_squared_error(t, p) ** 0.5,
    "mae": skm.mean_absolute_error,
    "mape": skm.mean_absolute_percentage_error,
    "r2": skm.r2_score,
    "accuracy": skm.accuracy_score,
    "f1": lambda t, p: skm.f1_score(t, p, average="binary"),
    "macro_f1": lambda t, p: skm.f1_score(t, p, average="macro"),
    "weighted_f1": lambda t, p: skm.f1_score(t, p, average="weighted"),
    "roc_auc": skm.roc_auc_score,   # ждёт вероятности
    "logloss": skm.log_loss,        # ждёт вероятности
}
LOWER_IS_BETTER = {"rmse", "mae", "mape", "logloss"}
NEEDS_PROBA = {"roc_auc", "logloss"}

AG_METRIC = {
    "rmse": "rmse", "mae": "mae", "mape": "mape", "r2": "r2",
    "accuracy": "accuracy", "f1": "f1", "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted", "roc_auc": "roc_auc", "logloss": "log_loss",
}
AG_PROBLEM = {"regression": "regression", "binary": "binary", "classification": "multiclass"}


def compute_metric(y_true, y_pred, metric):
    return float(METRICS[metric](y_true, y_pred))


def is_better(new_score, old_score, metric):
    if old_score is None:
        return True
    return new_score < old_score if metric in LOWER_IS_BETTER else new_score > old_score


def _prepare(task):
    train = pd.read_csv(task["train_path"])
    test = pd.read_csv(task["test_path"])
    y = train[task["target_col"]]
    X = train.drop(columns=[task["id_col"], task["target_col"]])
    X_test = test.drop(columns=[task["id_col"]])

    n = len(X)
    both = pd.concat([X, X_test], axis=0)
    # флаг "пропущено" для каждой колонки с пропусками: универсальный признак
    for col in [c for c in both.columns if both[c].isna().any()]:
        both[col + "_missing"] = both[col].isna().astype("int8")
    for col in both.select_dtypes(exclude="number").columns:
        both[col] = both[col].astype("category")
    X, X_test = both.iloc[:n], both.iloc[n:]

    stratify = y if task["task_type"] != "regression" else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    return X_train, X_valid, y_train, y_valid, X_test


def _predict(model, X, task, log_target):
    if task["metric"] in NEEDS_PROBA:
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        return proba[:, 1] if task["task_type"] == "binary" else proba
    pred = np.asarray(model.predict(X))
    if task["task_type"] == "regression" and log_target:
        pred = np.expm1(pred)
    return pred


def _fit_lgbm(X_train, y_train, task, params):
    log_target = params.get("log_target", False)
    if task["task_type"] == "regression":
        model = LGBMRegressor(verbose=-1)
        model.fit(X_train, np.log1p(y_train) if log_target else y_train)
    else:
        model = LGBMClassifier(verbose=-1, class_weight=params.get("class_weight"))
        model.fit(X_train, y_train)
    return model


def _fit_autogluon(X_train, y_train, task, params, time_budget):
    from autogluon.tabular import TabularPredictor

    log_target = params.get("log_target", False)
    train_df = X_train.copy()
    train_df[task["target_col"]] = np.log1p(y_train) if log_target else y_train
    predictor = TabularPredictor(
        label=task["target_col"],
        problem_type=AG_PROBLEM[task["task_type"]],
        eval_metric=AG_METRIC[task["metric"]],
        verbosity=0,
    )
    predictor.fit(
        train_df,
        time_limit=time_budget or 900,
        presets=params.get("presets", "medium_quality"),
    )
    return predictor


def train_and_predict(task, params, time_budget=None):
    X_train, X_valid, y_train, y_valid, X_test = _prepare(task)
    log_target = params.get("log_target", False) and task["task_type"] == "regression"

    if params.get("model", "lgbm") == "autogluon":
        model = _fit_autogluon(X_train, y_train, task, params, time_budget)
    else:
        model = _fit_lgbm(X_train, y_train, task, params)

    score = compute_metric(y_valid, _predict(model, X_valid, task, log_target), task["metric"])
    predictions = _predict(model, X_test, task, log_target)
    return {"score": score, "predictions": predictions.tolist()}


def _find_csv(task_dir, name, task_text):
    import re
    for rel in re.findall(r"[`'\"]?([\w./\-]+/" + name + r")[`'\"]?", task_text):
        p = task_dir / rel
        if p.exists():
            return p
    hits = sorted(task_dir.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} не найден рядом с {task_dir}")
    return hits[0]


def _detect_metric(task_text, task_type):
    t = task_text.lower()
    order = [("roc_auc", ["roc-auc", "roc_auc", "auc"]), ("logloss", ["log loss", "logloss", "log_loss"]),
             ("macro_f1", ["macro f1", "macro-f1", "f1_macro", "average=\"macro\""]),
             ("weighted_f1", ["weighted f1", "f1_weighted"]), ("f1", ["f1"]),
             ("accuracy", ["accuracy"]), ("rmse", ["rmse"]), ("mape", ["mape"]),
             ("mae", ["mae"]), ("r2", ["r2", "r^2", "r²"])]
    for name, keys in order:
        if any(k in t for k in keys):
            return name
    return "rmse" if task_type == "regression" else "macro_f1"


def build_task(task_md_path):
    from pathlib import Path
    task_md_path = Path(task_md_path).resolve()
    task_dir = task_md_path.parent
    text = task_md_path.read_text(encoding="utf-8") if task_md_path.exists() else ""

    train_path = _find_csv(task_dir, "train.csv", text)
    test_path = _find_csv(task_dir, "test.csv", text)
    sample_path = _find_csv(task_dir, "sample_submission.csv", text)

    sample = pd.read_csv(sample_path, nrows=50)
    id_col, target_col = sample.columns[0], sample.columns[1]

    y = pd.read_csv(train_path, usecols=[target_col])[target_col]
    n_unique = y.nunique()
    if y.dtype == object or str(y.dtype) == "string" or (n_unique <= 20 and (y.dropna() % 1 == 0).all()):
        task_type = "binary" if n_unique == 2 else "classification"
    else:
        task_type = "regression"

    metric = _detect_metric(text, task_type)
    s = sample[target_col]
    if task_type != "regression" and metric not in NEEDS_PROBA and pd.api.types.is_float_dtype(s) and (s.dropna() % 1 != 0).any():
        metric = "roc_auc" if task_type == "binary" else "logloss"

    return {
        "train_path": str(train_path), "test_path": str(test_path),
        "id_col": id_col, "target_col": target_col,
        "task_type": task_type, "metric": metric,
    }


def write_submission(task, predictions, out_path):
    test_ids = pd.read_csv(task["test_path"], usecols=[task["id_col"]])[task["id_col"]]
    if len(predictions) != len(test_ids):
        raise ValueError(f"predictions {len(predictions)} != test {len(test_ids)}")
    pd.DataFrame({task["id_col"]: test_ids, task["target_col"]: predictions}).to_csv(out_path, index=False)


def solve(task, out_path, time_budget=7200, log=print):
    import time
    t0 = time.time()
    stages = [
        ("lgbm", {}, None),
        ("lgbm_alt", {"log_target": True} if task["task_type"] == "regression"
                     else {"class_weight": "balanced"}, None),
        ("autogluon_medium", {"model": "autogluon", "presets": "medium_quality"}, 0.3),
        ("autogluon_good", {"model": "autogluon", "presets": "good_quality"}, 1.0),
    ]
    best, history = None, []
    for name, params, share in stages:
        left = time_budget - (time.time() - t0)
        if left < 60:
            break
        budget = int(left * share) if share else None
        try:
            res = train_and_predict(task, params, time_budget=budget)
            improved = is_better(res["score"], best, task["metric"])
            if improved:
                write_submission(task, res["predictions"], out_path)
                best = res["score"]
            history.append({"stage": name, "params": params, "score": res["score"], "kept": improved})
            log(f"[solve] {name:17s} score={res['score']:.5f} kept={improved} elapsed={time.time()-t0:.0f}s")
        except Exception as e:
            history.append({"stage": name, "params": params, "error": repr(e)})
            log(f"[solve] {name:17s} FAILED: {e!r}")
    return history


if __name__ == "__main__":
    import sys, time

    tasks = {
        "regression": {
            "train_path": "data/regression/train.csv", "test_path": "data/regression/test.csv",
            "id_col": "id", "target_col": "site_eui", "task_type": "regression", "metric": "rmse",
        },
        "classification": {
            "train_path": "data/multiclass/train.csv", "test_path": "data/multiclass/test.csv",
            "id_col": "id", "target_col": "attack_cat", "task_type": "classification", "metric": "macro_f1",
        },
    }

    use_ag = len(sys.argv) > 1 and sys.argv[1] == "autogluon"
    for name, task in tasks.items():
        t0 = time.time()
        params = {"model": "autogluon"} if use_ag else {}
        result = train_and_predict(task, params, time_budget=180)
        print(f"{name:14s} {params.get('model','lgbm'):9s} score={result['score']:.4f} "
              f"n={len(result['predictions'])} time={time.time()-t0:.0f}s")
