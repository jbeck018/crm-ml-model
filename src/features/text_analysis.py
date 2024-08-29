from transformers import pipeline
import pandas as pd

class TextAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")

    def analyze_sentiment(self, text: str) -> float:
        result = self.sentiment_analyzer(text)[0]
        return result['score'] if result['label'] == 'POSITIVE' else -result['score']

    def summarize_text(self, text: str, max_length: int = 100) -> str:
        return self.summarizer(text, max_length=max_length)[0]['summary_text']

    def process_text_data(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        df[f'{text_column}_sentiment'] = df[text_column].apply(self.analyze_sentiment)
        df[f'{text_column}_summary'] = df[text_column].apply(self.summarize_text)
        return df