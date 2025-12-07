import pandas as pd

from utils import load_data, temporal_split, get_xy, evaluate_model
from logistic_regression import build_logistic_regression
from random_forest import build_random_forest
from gradient_boosting import build_gradient_boosting


FEATURE_COLS = ["home_spread", "over_under_line"]
TARGET_COL = "_home_win"


def train_and_eval_model(name: str, model, X_train, y_train, X_val, y_val):
    """
    Fit model on training data, evaluate on validation data,
    return metrics dict and fitted model.
    """
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_val_proba = model.predict_proba(X_val)[:, 1]
    else:
        y_val_proba = None

    metrics = evaluate_model(name, y_val, y_val_pred, y_val_proba)
    return metrics, model


def main():
    # 1) Load & split data
    df = load_data(min_season=2002)
    train_df, val_df, test_df = temporal_split(df)

    X_train, y_train = get_xy(train_df, FEATURE_COLS, TARGET_COL)
    X_val,   y_val   = get_xy(val_df,   FEATURE_COLS, TARGET_COL)
    X_test,  y_test  = get_xy(test_df,  FEATURE_COLS, TARGET_COL)

    # 2) Build each model
    models = {
        "Logistic Regression": build_logistic_regression(),
        "Random Forest":       build_random_forest(),
        "Gradient Boosting":   build_gradient_boosting(),
    }

    # 3) Train + evaluate each on validation set
    results = []
    fitted_models = {}

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training {name}...")
        metrics, fitted = train_and_eval_model(name, model, X_train, y_train, X_val, y_val)
        results.append(metrics)
        fitted_models[name] = fitted

    results_df = pd.DataFrame(results)
    print("\n\n=== Validation Performance Summary ===")
    print(results_df.sort_values(by="roc_auc", ascending=False))

    # 4) Choose best model by ROC-AUC
    best_row = results_df.sort_values(by="roc_auc", ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nBest model on validation set: {best_name}")

    # 5) Retrain best model on Train + Val, evaluate on Test
    train_val_df = pd.concat([train_df, val_df], axis=0)
    X_train_val, y_train_val = get_xy(train_val_df, FEATURE_COLS, TARGET_COL)

    # Rebuild a fresh instance of the best model then fit on train+val
    if best_name == "Logistic Regression":
        best_model = build_logistic_regression()
    elif best_name == "Random Forest":
        best_model = build_random_forest()
    else:
        best_model = build_gradient_boosting()

    best_model.fit(X_train_val, y_train_val)

    y_test_pred = best_model.predict(X_test)
    if hasattr(best_model, "predict_proba"):
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = None

    test_metrics = evaluate_model(best_name + " (Train+Val → Test)", y_test, y_test_pred, y_test_proba)

    print("\n=== Final Test Performance ===")
    print(test_metrics)

    # 6) Save validation comparison for your report / appendix
    results_df.to_csv("sprint3_validation_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
