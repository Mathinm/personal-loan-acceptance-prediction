"""Train all models, evaluate on the held-out test set, save results + best model.

Two experiments:
  1. Baseline  - replicates the reference paper (Perceptron, Logistic Regression,
                 K-Means, KNN, linear SVM, MLP) with default settings, unscaled
                 features (MLP scaled, as in the paper), no imbalance handling.
  2. Improved  - every model in a Pipeline: StandardScaler -> (SMOTE) -> classifier,
                 class imbalance handled, hyper-parameters tuned with 5-fold
                 stratified CV (scoring = F1), plus Random Forest and XGBoost.

Run from the project root:  python src/train.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import (compute_metrics, plot_confusion_grid,  # noqa: E402
                      plot_feature_importance, plot_roc)
from preprocess import RANDOM_STATE, get_data, scale  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """K-Means (k=2) used as a classifier: each cluster is mapped to the majority
    label of the training points inside it (unsupervised baseline from the paper)."""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X, y):
        self.km_ = KMeans(n_clusters=2, n_init=10, random_state=self.random_state).fit(X)
        y = np.asarray(y)
        self.map_ = {c: int(round(y[self.km_.labels_ == c].mean())) for c in range(2)}
        self.pos_cluster_ = max(range(2), key=lambda c: y[self.km_.labels_ == c].mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.array([self.map_[c] for c in self.km_.predict(X)])

    def decision_function(self, X):
        d = self.km_.transform(X)                     # distance to each centroid
        return d[:, 1 - self.pos_cluster_] - d[:, self.pos_cluster_]


def run_baseline(X_train, X_test, y_train, y_test):
    print("\n=== Experiment 1: Baseline (paper replication) ===")
    X_train_s, X_test_s, _ = scale(X_train, X_test)
    raw = {
        "Perceptron": Perceptron(random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, max_iter=2000),
    }
    fitted, rows = {}, []
    for name, m in raw.items():
        m.fit(X_train, y_train)
        fitted[name] = (m, X_test)
    for name, m in {
        "K-Means (k=2)": KMeansClassifier(),
        "MLP (100 neurons)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                                           random_state=RANDOM_STATE),
    }.items():
        m.fit(X_train_s, y_train)
        fitted[name] = (m, X_test_s)
    for name, (m, Xt) in fitted.items():
        rows.append(compute_metrics(name, m, Xt, y_test))
    df = pd.DataFrame(rows).set_index("Model").round(4)
    print(df.to_string())
    plot_roc(fitted, None, y_test, "ROC - Baseline (paper replication)", "roc_baseline.png")
    plot_confusion_grid(fitted, y_test, "Confusion matrices - Baseline", "cm_baseline.png")
    return df


def improved_search_spaces():
    scaler = ("scaler", StandardScaler())
    smote = ("smote", SMOTE(random_state=RANDOM_STATE))
    return {
        "Perceptron": (
            Pipeline([scaler, ("clf", Perceptron(class_weight="balanced", random_state=RANDOM_STATE))]),
            {"clf__penalty": [None, "l2"], "clf__alpha": [1e-4, 1e-3, 1e-2]},
        ),
        "Logistic Regression": (
            Pipeline([scaler, ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))]),
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "KNN + SMOTE": (
            ImbPipeline([scaler, smote, ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]},
        ),
        "SVM": (
            Pipeline([scaler, ("clf", SVC(class_weight="balanced", probability=True,
                                          random_state=RANDOM_STATE))]),
            {"clf__kernel": ["linear", "rbf"], "clf__C": [0.1, 1, 10]},
        ),
        "MLP + SMOTE": (
            ImbPipeline([scaler, smote, ("clf", MLPClassifier(max_iter=800, early_stopping=True,
                                                              random_state=RANDOM_STATE))]),
            {"clf__hidden_layer_sizes": [(64,), (100,), (64, 32)], "clf__alpha": [1e-4, 1e-3]},
        ),
        "Random Forest": (
            Pipeline([scaler, ("clf", RandomForestClassifier(class_weight="balanced",
                                                             random_state=RANDOM_STATE, n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10],
             "clf__min_samples_leaf": [1, 2]},
        ),
        "XGBoost": (
            Pipeline([scaler, ("clf", XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                                                    n_jobs=-1))]),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5],
             "clf__learning_rate": [0.05, 0.1], "clf__scale_pos_weight": [1, 9.4]},
        ),
    }


def run_improved(X_train, X_test, y_train, y_test):
    print("\n=== Experiment 2: Improved (scaling + imbalance handling + tuning) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows, params = {}, [], {}
    for name, (pipe, grid) in improved_search_spaces().items():
        t = time.time()
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        fitted[name] = (best, X_test)
        r = compute_metrics(name, best, X_test, y_test)
        r["CV_F1"] = gs.best_score_
        rows.append(r)
        params[name] = {k.replace("clf__", ""): (list(v) if isinstance(v, tuple) else v)
                        for k, v in gs.best_params_.items()}
        print(f"  {name:20s} CV-F1={gs.best_score_:.4f}  ({time.time() - t:.1f}s)  {params[name]}")
    df = pd.DataFrame(rows).set_index("Model").round(4)
    print(df.to_string())
    plot_roc(fitted, None, y_test, "ROC - Improved models", "roc_improved.png")
    plot_confusion_grid(fitted, y_test, "Confusion matrices - Improved", "cm_improved.png")
    return df, fitted, params


def main():
    X_train, X_test, y_train, y_test = get_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, "
          f"positive rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    base = run_baseline(X_train, X_test, y_train, y_test)
    base.to_csv(RESULTS / "metrics_baseline.csv")

    imp, fitted, params = run_improved(X_train, X_test, y_train, y_test)
    imp.to_csv(RESULTS / "metrics_improved.csv")
    with open(RESULTS / "best_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Best model chosen on cross-validated F1 (not on the test set)
    best_name = imp["CV_F1"].idxmax()
    best_model = fitted[best_name][0]
    joblib.dump({"model": best_model, "features": list(X_train.columns), "name": best_name},
                MODELS / "best_model.pkl")
    print(f"\nBest model (by CV F1): {best_name} -> saved to models/best_model.pkl")

    for name in ["Random Forest", "XGBoost"]:
        clf = fitted[name][0].named_steps["clf"]
        fname = "feature_importance_" + name.lower().replace(" ", "_") + ".png"
        imp_s = plot_feature_importance(clf, X_train.columns, f"Feature importance - {name}", fname)
        imp_s.round(4).to_csv(RESULTS / fname.replace(".png", ".csv"), header=["importance"])

    # Combined comparison table
    comp = pd.concat({"Baseline": base[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]],
                      "Improved": imp[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]})
    comp.to_csv(RESULTS / "metrics_all.csv")


if __name__ == "__main__":
    main()
