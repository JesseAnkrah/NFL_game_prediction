from sklearn.ensemble import GradientBoostingClassifier


def build_gradient_boosting():
    """
    Gradient Boosting classifier (GBM-style).
    Slower than RF, but often better calibrated and more accurate.
    """
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    return model
