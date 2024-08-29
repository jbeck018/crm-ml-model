from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def recursive_feature_elimination(X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 20):
    rfe_selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=n_features_to_select, step=1)
    rfe_selector = rfe_selector.fit(X, y)
    return X.columns[rfe_selector.support_].tolist()

def feature_importance_selection(X: pd.DataFrame, y: pd.Series, threshold: float = 'median'):
    selector = SelectFromModel(RandomForestClassifier(), threshold=threshold)
    selector = selector.fit(X, y)
    return X.columns[selector.get_support()].tolist()