from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from ..models import EnsembleModel
from ..utils.helpers import load_model, load_config
from ..utils.explanation_agent import ExplanationAgent

app = FastAPI()

config = load_config("config.yaml")

# Load pre-trained ensemble model
ensemble_model = load_model(EnsembleModel, "models/ensemble/best_model.pth", config['models'])

explanation_agent = ExplanationAgent(config['openai_api_key'])
# Load account data and create database (this should be done periodically to keep data up-to-date)
account_data = load_account_data()  # Implement this function to load your account data
explanation_agent.create_account_database(account_data)

class PredictionRequest(BaseModel):
    account_id: str
    features: dict

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_explanation: str
    health_score: float
    health_explanation: str
    expansion_probability: float
    expansion_explanation: str
    upsell_opportunities: list
    upsell_explanation: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        input_df = pd.DataFrame([request.features])
        
        # Make predictions using ensemble model
        probabilities = ensemble_model.predict_proba(input_df.values)
        churn_prob, health_score, expansion_prob, *upsell_probs = probabilities[0]

        # Generate explanations
        churn_explanation = explanation_agent.generate_explanation("Churn", churn_prob, get_top_features(ensemble_model, input_df), request.account_id)
        health_explanation = explanation_agent.generate_explanation("Health", health_score, get_top_features(ensemble_model, input_df), request.account_id)
        expansion_explanation = explanation_agent.generate_explanation("Expansion", expansion_prob, get_top_features(ensemble_model, input_df), request.account_id)
        upsell_explanation = explanation_agent.generate_explanation("Upsell", max(upsell_probs), get_top_features(ensemble_model, input_df), request.account_id)

        return PredictionResponse(
            churn_probability=churn_prob,
            churn_explanation=churn_explanation,
            health_score=health_score,
            health_explanation=health_explanation,
            expansion_probability=expansion_prob,
            expansion_explanation=expansion_explanation,
            upsell_opportunities=upsell_probs,
            upsell_explanation=upsell_explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_top_features(model, input_df, n=5):
    feature_importance = model.get_feature_importance(input_df)
    return feature_importance.nlargest(n).index.tolist()