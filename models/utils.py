from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
)


# ---------- DATA LOADING & SPLITTING ----------

def load_data(min_season: int = 2002) -> pd.DataFrame:
    """
    Load the cleaned modeling dataset.

    Assumes scores_model.csv is in the project root, one level above /models.
    If yours is in data/scores_model.csv, change the path below.
    """
    root = Path(__file__).resolve().parents[1]
    csv_path = root /  "data" / "processed" / "scores_model.csv"
    df = pd.read_csv(csv_path)

    required_cols = ["schedule_season", "home_spread", "over_under_line", "_home_win"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in scores_model.csv: {missing}")

    # Keep modern era where stats & markets are consistent
    df = df[df["schedule_season"] >= min_season].copy()

    # Drop rows with missing key features / target
    df = df.dropna(subset=["home_spread", "over_under_line", "_home_win"])

    df["_home_win"] = df["_home_win"].astype(int)

    return df


def temporal_split(df: pd.DataFrame):
    """
    Temporal split based on schedule_season:

        Train: seasons <= 2018
        Val:   seasons 2019–2021
        Test:  seasons >= 2022
    """
    train_df = df[df["schedule_season"] <= 2018].copy()
    val_df   = df[(df["schedule_season"] >= 2019) & (df["schedule_season"] <= 2021)].copy()
    test_df  = df[df["schedule_season"] >= 2022].copy()

    print("Train seasons:", train_df["schedule_season"].min(), "to", train_df["schedule_season"].max())
    print("Val seasons:  ", val_df["schedule_season"].min(), "to", val_df["schedule_season"].max())
    print("Test seasons: ", test_df["schedule_season"].min(), "to", test_df["schedule_season"].max())

    return train_df, val_df, test_df


def get_xy(df: pd.DataFrame, feature_cols, target_col: str = "_home_win"):
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


# ---------- METRICS / EVALUATION ----------

def evaluate_model(name: str, y_true, y_pred, y_proba=None) -> dict:
    """
    Compute standard classification metrics and print a quick summary.
    Returns a dict you can collect into a DataFrame.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        try:
            roc  = roc_auc_score(y_true, y_proba)
        except ValueError:
            roc = np.nan
        brier = brier_score_loss(y_true, y_proba)
    else:
        roc = np.nan
        brier = np.nan

    print(f"\n=== {name} ===")
    print("Accuracy:    ", round(acc, 3))
    print("Precision:   ", round(prec, 3))
    print("Recall:      ", round(rec, 3))
    print("F1-score:    ", round(f1, 3))
    print("ROC-AUC:     ", "N/A" if np.isnan(roc) else round(roc, 3))
    print("Brier score: ", "N/A" if np.isnan(brier) else round(brier, 3))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification report:\n",
          classification_report(y_true, y_pred, zero_division=0))

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "brier": brier,
    }
