import pandas as pd
from typing import List, Dict, Any
from transformers import pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.scaler = StandardScaler()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_usage_metrics(df)
        df = self.analyze_text_sentiment(df)
        df = self.encode_categorical_variables(df)
        df = self.create_interaction_features(df)
        df = self.select_top_features(df)
        return df

    def calculate_usage_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        usage_columns = [col for col in df.columns if 'usage' in col.lower() or 'calls' in col.lower()]
        df['total_usage'] = df[usage_columns].sum(axis=1)
        df['usage_growth'] = df['total_usage'].pct_change()
        return df

    def analyze_text_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            df[f'{col}_sentiment'] = df[col].apply(
                lambda x: self.sentiment_analyzer(x)[0]['score'] if pd.notnull(x) else None
            )
        return df

    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        return pd.get_dummies(df, columns=categorical_columns, dummy_na=True)

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        return df

    def select_top_features(self, df: pd.DataFrame, target_col: str, n_features: int = 50) -> pd.DataFrame:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        mi_scores = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        top_features = mi_scores.head(n_features).index.tolist()
        return df[top_features + [target_col]]

    def normalize_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df