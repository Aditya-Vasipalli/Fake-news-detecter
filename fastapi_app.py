# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# from newspaper import Article
# import shap

# # Create FastAPI app
# app = FastAPI()

# # Load the saved model and vectorizer
# model = joblib.load("model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # SHAP explainer
# explainer = shap.Explainer(model.predict, vectorizer.transform)

# # Input schema using Pydantic
# class ArticleRequest(BaseModel):
#     text: str

# class URLRequest(BaseModel):
#     url: str

# # Health check endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Fake News Classifier is running."}

# # Prediction endpoint for text
# @app.post("/predict")
# def predict(request: ArticleRequest):
#     # Vectorize the input text
#     vectorized = vectorizer.transform([request.text])
    
#     # Predict
#     prediction = model.predict(vectorized)[0]
#     probability = model.predict_proba(vectorized)[0]

#     # SHAP explanation
#     shap_values = explainer(vectorized)
#     feature_names = vectorizer.get_feature_names_out()
#     shap_values_dense = shap_values.values[0]
#     top_indices = shap_values_dense.argsort()[-10:][::-1]  # Top 10 features
#     top_features = [
#         {"word": feature_names[i], "shap_value": shap_values_dense[i]}
#         for i in top_indices
#     ]

#     return {
#         "prediction": "Fake" if prediction == 1 else "Real",
#         "confidence": {
#             "real": round(probability[0], 2),
#             "fake": round(probability[1], 2)
#         },
#         "shap_explanation": top_features
#     }

# # Prediction endpoint for URLs
# @app.post("/predict-url")
# def predict_url(request: URLRequest):
#     # Fetch and parse the article
#     article = Article(request.url)
#     try:
#         article.download()
#         article.parse()
#     except Exception as e:
#         raise HTTPException(status_code=400, detail="Failed to fetch or parse the article.")

#     # Combine the title and text
#     text = article.title + " " + article.text

#     # Vectorize the article text
#     vectorized = vectorizer.transform([text])
    
#     # Predict
#     prediction = model.predict(vectorized)[0]
#     probability = model.predict_proba(vectorized)[0]

#     return {
#         "title": article.title,
#         "prediction": "Fake" if prediction == 1 else "Real",
#         "confidence": {
#             "real": round(probability[0], 2),
#             "fake": round(probability[1], 2)
#         }
#     }

# @app.get("/test-shap")
# def test_shap():
#     sample_text = "This is a sample news article."
#     vectorized = vectorizer.transform([sample_text])
#     shap_values = explainer(vectorized)
    
#     # Debugging: Print the SHAP values structure
#     print(shap_values)
    
#     return {"message": "Check the server logs for SHAP values structure."}
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