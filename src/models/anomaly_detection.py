from sklearn.ensemble import IsolationForest
import pandas as pd

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def detect_anomalies(self, X: pd.DataFrame):
        anomaly_labels = self.predict(X)
        anomalies = X[anomaly_labels == -1]
        return anomalies