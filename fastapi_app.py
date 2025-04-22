from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from newspaper import Article

# Create FastAPI app
app = FastAPI()

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Input schema using Pydantic
class ArticleRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Fake News Classifier is running."}

# Prediction endpoint for text
@app.post("/predict")
def predict(request: ArticleRequest):
    # Vectorize the input text
    vectorized = vectorizer.transform([request.text])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": {
            "real": round(probability[0], 2),
            "fake": round(probability[1], 2)
        }
    }

# Prediction endpoint for URLs
@app.post("/predict-url")
def predict_url(request: URLRequest):
    # Fetch and parse the article
    article = Article(request.url)
    try:
        article.download()
        article.parse()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to fetch or parse the article.")

    # Combine the title and text
    text = article.title + " " + article.text

    # Vectorize the article text
    vectorized = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    return {
        "title": article.title,
        "prediction": "Fake" if prediction == 1 else "Real",
        "confidence": {
            "real": round(probability[0], 2),
            "fake": round(probability[1], 2)
        }
    }