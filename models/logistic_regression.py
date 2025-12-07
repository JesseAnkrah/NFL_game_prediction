from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_logistic_regression():
    """
    Logistic Regression pipeline with standardization.
    Simple, interpretable baseline for binary classification.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",  # good for small feature sets
            penalty="l2",
            C=1.0,
            max_iter=1000,
        ))
    ])
    return model
