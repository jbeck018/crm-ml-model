from .base_model import BaseModel
from .stacking_ensemble import StackingEnsemble
import torch.nn as nn

class HealthModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.ensemble = self._create_ensemble()

    def _create_ensemble(self):
        base_models = [
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(3)  # Create 3 base models
        ]
        return StackingEnsemble(base_models, config=self.config)

    def forward(self, x):
        return self.ensemble(x)

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    @classmethod
    def load_from_checkpoint(cls, path, config):
        model = cls(config)
        model.load_state_dict(torch.load(path))
        return torch.compile(model)