import yaml
import torch
from typing import Dict, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss
import numpy as np

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model_class, path: str, config: Dict[str, Any]):
    model = model_class(config)
    model.load_state_dict(torch.load(path))
    return torch.compile(model)  # Compile the model for better performance

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_prob is not None:
        if y_prob.ndim == 2:
            # Multi-class case
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            metrics['pr_auc'] = average_precision_score(y_true, y_prob, average='weighted')
        else:
            # Binary case
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        
        # Add log loss if probabilities are available
        try:
            metrics['log_loss'] = log_loss(y_true, y_prob)
        except ValueError:
            # This can happen if y_prob contains probabilities only for the positive class
            # In this case, we'll compute log loss manually
            eps = 1e-15
            y_prob = np.clip(y_prob, eps, 1 - eps)
            metrics['log_loss'] = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    # Calculate confidence intervals for accuracy
    n = len(y_true)
    z = 1.96  # 95% confidence interval
    accuracy = metrics['accuracy']
    ci = z * np.sqrt((accuracy * (1 - accuracy)) / n)
    metrics['accuracy_95ci'] = (accuracy - ci, accuracy + ci)

    return metrics