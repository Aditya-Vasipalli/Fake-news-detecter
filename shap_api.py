import shap
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from newspaper import Article
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize FastAPI app
app = FastAPI(title="Fake News Detector API with SHAP")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["chrome-extension://<your-extension-id>"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prepare a dummy explainer (required once)
dummy_array = vectorizer.transform(["test input"]).toarray()
dummy_df = pd.DataFrame(dummy_array, columns=vectorizer.get_feature_names_out())
explainer = shap.Explainer(model, dummy_df)

# Response schema
class PredictionResponse(BaseModel):
    title: str
    prediction: str
    confidence_real: float
    confidence_fake: float
    top_keywords: list

@app.get("/predict", response_model=PredictionResponse)
def predict_article(url: str = Query(..., description="URL of the news article")):
    try:
        # Fetch and parse article
        article = Article(url)
        article.download()
        article.parse()
        text = article.title + " " + article.text

        # Transform input
        article_vectorized = vectorizer.transform([text])
        prediction = model.predict(article_vectorized)[0]
        prediction_proba = model.predict_proba(article_vectorized)[0]

        # Convert to DataFrame for SHAP explainer
        article_array = article_vectorized.toarray()
        article_df = pd.DataFrame(article_array, columns=vectorizer.get_feature_names_out())

        # Get SHAP values using the DataFrame
        shap_values = explainer(article_df)
        values = shap_values[0].values
        base_value = shap_values.base_values[0]

        # Only show non-zero features
        indices = article_vectorized.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        words = feature_names[indices]
        vals = values[indices]
        data_vals = article_array[0][indices]

        # Build a custom SHAP Explanation with word labels
        explanation = shap.Explanation(
            values=vals,
            base_values=base_value,
            data=data_vals,
            feature_names=words
        )

        # Save SHAP bar plot with words
        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig("shap_plot.png")
        plt.close()

        # Top 10 words by absolute SHAP value
        top_n = 10
        top_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
        top_keywords = [
            {
                "word": words[i],
                "shap_value": float(vals[i]),
                "direction": "Fake ⬆️" if vals[i] > 0 else "Real ⬇️"
            }
            for i in top_idx
        ]

        return {
            "title": article.title,
            "prediction": "Fake" if prediction == 1 else "Real",
            "confidence_real": round(prediction_proba[0], 2),
            "confidence_fake": round(prediction_proba[1], 2),
            "top_keywords": top_keywords
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict", response_model=PredictionResponse)
def predict_article_post(payload: dict = Body(...)):
    url = payload.get("url")
    if not url:
        return JSONResponse(status_code=400, content={"error": "No URL provided"})
    return predict_article(url)

@app.get("/plot")
def get_shap_plot():
    if os.path.exists("shap_plot.png"):
        return FileResponse("shap_plot.png", media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Plot not available"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
