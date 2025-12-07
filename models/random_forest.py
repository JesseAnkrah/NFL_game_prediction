from sklearn.ensemble import RandomForestClassifier


def build_random_forest():
    """
    Random Forest classifier to capture non-linear relationships.
    Hyperparameters are reasonable defaults you can tune later.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    return model
