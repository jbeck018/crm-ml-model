from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def tune_random_forest(X, y, param_distributions, n_iter=100, cv=5):
    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(rf, param_distributions, n_iter=n_iter, cv=cv, scoring='roc_auc', n_jobs=-1)
    random_search.fit(X, y)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best ROC AUC score: {random_search.best_score_:.3f}")
    return random_search.best_estimator_

# Example usage:
# param_distributions = {
#     'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
#     'max_features': ['auto', 'sqrt'],
#     'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
# best_rf = tune_random_forest(X, y, param_distributions)