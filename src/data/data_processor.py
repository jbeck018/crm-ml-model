import pandas as pd
from typing import Dict, Any
from src.features.feature_engineering import FeatureEngineer
from src.models.anomaly_detection import AnomalyDetector
from sklearn.impute import SimpleImputer, KNNImputer

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.anomaly_detector = AnomalyDetector(contamination=config.get('anomaly_contamination', 0.1))

    def process_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        df = self.handle_missing_values(df)
        df = self.feature_engineer.engineer_features(df)
        df = self.detect_and_handle_anomalies(df)
        df = self.feature_engineer.select_top_features(df, target_col)
        df = self.feature_engineer.normalize_numerical_features(df)
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns

        cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

        knn_imputer = KNNImputer(n_neighbors=5)
        df[numeric_columns] = knn_imputer.fit_transform(df[numeric_columns])

        return df

    def detect_and_handle_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        self.anomaly_detector.fit(df)
        anomalies = self.anomaly_detector.detect_anomalies(df)
        
        # Log anomalies for further investigation
        print(f"Detected {len(anomalies)} anomalies")
        
        # For now, we'll keep the anomalies in the dataset
        # In a real-world scenario, you might want to handle them differently
        # e.g., remove them, or create a flag for anomalous data points
        df['is_anomaly'] = self.anomaly_detector.predict(df)
        
        return df