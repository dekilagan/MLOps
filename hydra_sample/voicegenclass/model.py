# model.py

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_model(cfg):
    """
    Instantiate and return a model based on the configuration.
    """
    model_type = cfg.model.type.lower()
    if model_type == "svc":
        model = SVC(kernel=cfg.model.svc.kernel, C=cfg.model.svc.C, gamma=cfg.model.svc.gamma)
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg.model.random_forest.n_estimators,
            max_depth=cfg.model.random_forest.max_depth,
            random_state=cfg.model.random_state
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            penalty=cfg.model.logistic_regression.penalty,
            C=cfg.model.logistic_regression.C,
            solver=cfg.model.logistic_regression.solver,
            max_iter=1000,
            random_state=cfg.model.random_state
        )
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=cfg.model.knn.n_neighbors)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
    return model
