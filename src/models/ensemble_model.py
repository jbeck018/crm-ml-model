import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from .churn_model import ChurnModel
from .health_model import HealthModel
from .expansion_model import ExpansionModel
from .upsell_model import UpsellModel

class EnsembleModel:
    def __init__(self, config):
        self.config = config
        self.churn_model = ChurnModel(config)
        self.health_model = HealthModel(config)
        self.expansion_model = ExpansionModel(config)
        self.upsell_model = UpsellModel(config)

        self.ensemble = VotingClassifier(
            estimators=[
                ('churn', self.churn_model),
                ('health', self.health_model),
                ('expansion', self.expansion_model),
                ('upsell', self.upsell_model)
            ],
            voting='soft'
        )

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def get_feature_importance(self, X: pd.DataFrame) -> pd.Series:
        importances = []
        for name, model in self.ensemble.named_estimators_.items():
            importances.append(model.get_feature_importance(X))
        mean_importance = np.mean(importances, axis=0)
        return pd.Series(mean_importance, index=X.columns)