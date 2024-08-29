from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any

class StackingEnsemble(nn.Module):
    def __init__(self, base_models: List[nn.Module], meta_model=None, config: Dict[str, Any] = None):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model if meta_model else LogisticRegression()
        self.config = config or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_preds = torch.stack([model(x) for model in self.base_models], dim=1)
        return self.meta_model(base_preds)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        base_preds_train = np.column_stack([
            model(torch.tensor(X_train, dtype=torch.float32)).detach().numpy()
            for model in self.base_models
        ])
        
        base_preds_val = np.column_stack([
            model(torch.tensor(X_val, dtype=torch.float32)).detach().numpy()
            for model in self.base_models
        ])
        
        self.meta_model.fit(base_preds_train, y_train)
        
        ensemble_score = self.meta_model.score(base_preds_val, y_val)
        print(f"Ensemble validation score: {ensemble_score:.3f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([
            model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            for model in self.base_models
        ])
        return self.meta_model.predict(base_preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_preds = np.column_stack([
            model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
            for model in self.base_models
        ])
        return self.meta_model.predict_proba(base_preds)

    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        base_importances = [model.get_feature_importance(X) for model in self.base_models]
        return np.mean(base_importances, axis=0)