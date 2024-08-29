import torch
import torch.nn as nn
import pandas as pd
import shap
import numpy as np
from typing import Dict, Any
from src.data.data_processor import DataProcessor
from src.features.feature_selection import recursive_feature_elimination, feature_importance_selection
from src.utils.model_evaluation import evaluate_model, calculate_metrics
from src.utils.hyperparameter_tuning import tune_random_forest

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_processor = DataProcessor(config)
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.explainer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")

    def preprocess_data(self, data: pd.DataFrame, target_col: str) -> torch.Tensor:
        processed_data = self.data_processor.process_data(data, target_col)
        
        # Feature selection
        X = processed_data.drop(columns=[target_col])
        y = processed_data[target_col]
        selected_features = recursive_feature_elimination(X, y)
        selected_features += feature_importance_selection(X, y)
        selected_features = list(set(selected_features))  # Remove duplicates
        
        processed_data = processed_data[selected_features + [target_col]]
        
        return torch.tensor(processed_data.drop(columns=[target_col]).values, dtype=torch.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement predict method")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement predict_proba method")

    def get_feature_importance(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError("Subclasses must implement get_feature_importance method")

    def create_shap_explainer(self, data: pd.DataFrame, target_col: str):
        features = self.preprocess_data(data, target_col)
        self.explainer = shap.DeepExplainer(self, features)

    def get_shap_values(self, data: pd.DataFrame, target_col: str):
        features = self.preprocess_data(data, target_col)
        return self.explainer.shap_values(features)

    def get_global_shap_values(self, data: pd.DataFrame, target_col: str):
        shap_values = self.get_shap_values(data, target_col)
        return np.abs(shap_values).mean(0)

    def get_local_shap_values(self, data: pd.DataFrame, target_col: str):
        return self.get_shap_values(data, target_col)