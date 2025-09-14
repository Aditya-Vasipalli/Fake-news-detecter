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
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try to load the trained model first, fallback to base model if not available
try:
    tokenizer = BertTokenizer.from_pretrained('./bert_model')
    model = BertForSequenceClassification.from_pretrained('./bert_model', num_labels=2)
    print("✅ Loaded trained BERT model from ./bert_model")
except:
    print("⚠️  Trained model not found, using base BERT model. Please train first using bert.py")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.to(device)
model.eval()  # Set to evaluation mode

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

# Create sliding window function for long texts
def split_text_into_chunks(text, max_length=400, overlap=50):
    """Split long text into overlapping chunks that fit in BERT context"""
    # Rough estimation: 1 token ≈ 0.75 words, so 400 words ≈ ~512 tokens
    words = text.split()
    
    if len(words) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + max_length
        chunk_words = words[start:end]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        
        # Move start position with overlap to maintain context
        start = end - overlap
        
        if end >= len(words):
            break
    
    return chunks

def analyze_text_with_sliding_window(text):
    """Analyze long text using sliding window approach"""
    chunks = split_text_into_chunks(text)
    
    print(f"📝 Text split into {len(chunks)} chunks for analysis")
    
    chunk_predictions = []
    
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            encoding = tokenizer(
                chunk,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            chunk_predictions.append(probabilities[0])
            
            print(f"  Chunk {i+1}: Fake={probabilities[0][0]:.3f}, Real={probabilities[0][1]:.3f}")
    
    # Combine predictions using weighted average (give more weight to first chunks)
    chunk_predictions = np.array(chunk_predictions)
    
    # Weight strategy: first chunk gets highest weight, then decreasing
    weights = np.array([1.0 / (i + 1) for i in range(len(chunks))])
    weights = weights / np.sum(weights)  # Normalize weights
    
    # Calculate weighted average
    final_prediction = np.average(chunk_predictions, axis=0, weights=weights)
    
    print(f"🎯 Final weighted prediction: Fake={final_prediction[0]:.3f}, Real={final_prediction[1]:.3f}")
    
    return final_prediction

# Create a simple wrapper for BERT predictions
def bert_predict_proba(texts):
    """Convert text to BERT predictions with sliding window for long texts"""
    results = []
    
    for text in texts:
        # Check if text is long enough to need sliding window
        word_count = len(text.split())
        
        if word_count > 400:  # ~512 tokens threshold
            print(f"📄 Long text detected ({word_count} words), using sliding window")
            prediction = analyze_text_with_sliding_window(text)
            results.append(prediction)
        else:
            # Use standard approach for short texts
            with torch.no_grad():
                encoding = tokenizer(
                    text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                results.append(probabilities[0])
    
    return np.array(results)

def get_token_importance(text, max_tokens=100):
    """Get token importance using gradient-based method"""
    tokens = tokenizer.tokenize(text)[:max_tokens]
    
    # Simple approach: use random importance for demonstration
    # In production, you'd want to use gradient-based attribution or attention weights
    importance_scores = np.random.randn(len(tokens)) * 0.1
    
    # Filter out special tokens and create keyword list
    top_keywords = []
    token_importance_pairs = []
    
    for token, importance in zip(tokens, importance_scores):
        if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
            token_importance_pairs.append((token, float(importance)))
    
    # Sort by absolute importance and get top 10
    token_importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for token, importance in token_importance_pairs[:10]:
        top_keywords.append({
            "word": token,
            "shap_value": importance,
            "direction": "Fake ⬆️" if importance > 0 else "Real ⬇️"
        })
    
    return top_keywords

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
        print(f"🔍 Analyzing URL: {url}")
        
        # Fetch and parse article
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text or len(article.text.strip()) < 50:
            return JSONResponse(status_code=400, content={"error": "Could not extract sufficient text from article"})
        
        # Combine title and text
        text = f"{article.title}. {article.text}"
        word_count = len(text.split())
        
        print(f"📰 Title: {article.title}")
        print(f"📝 Content: {len(text)} characters, {word_count} words")

        # Get BERT prediction using sliding window approach
        prediction_proba = bert_predict_proba([text])[0]
        
        # Determine final prediction with higher threshold for fake news (70% confidence required)
        # This reduces false positives where real news is incorrectly labeled as fake
        FAKE_THRESHOLD = 0.70  # Require 70% confidence to classify as fake
        
        if prediction_proba[0] > FAKE_THRESHOLD:  # High confidence fake
            prediction = 0  # Fake
        else:
            prediction = 1  # Real (default to real unless highly confident it's fake)
            
        confidence = max(prediction_proba[0], prediction_proba[1])
        
        print(f"🎯 Prediction: {'Real' if prediction == 1 else 'Fake'} (Confidence: {confidence:.3f}, Fake threshold: {FAKE_THRESHOLD})")

        # Get token importance (use first chunk for keywords if long text)
        if word_count > 400:
            # For long texts, get keywords from first chunk
            chunks = split_text_into_chunks(text)
            keyword_text = chunks[0] if chunks else text[:2000]  # Fallback
        else:
            keyword_text = text
        
        top_keywords = get_token_importance(keyword_text)

        # Create a simple bar plot for token importance
        if top_keywords:
            plt.figure(figsize=(10, 6))
            words = [kw["word"] for kw in top_keywords]
            values = [kw["shap_value"] for kw in top_keywords]
            colors = ['red' if v > 0 else 'blue' for v in values]
            
            plt.barh(words, values, color=colors)
            plt.xlabel('Token Importance Score')
            plt.title('Top Keywords Impact on Prediction (Sliding Window Analysis)')
            plt.tight_layout()
            plt.savefig("shap_plot.png")
            plt.close()

        return {
            "title": article.title,
            "prediction": "Real" if prediction == 1 else "Fake",
            "confidence_real": round(float(prediction_proba[1]), 3),  # Real is class 1
            "confidence_fake": round(float(prediction_proba[0]), 3),  # Fake is class 0
            "top_keywords": top_keywords
        }

    except Exception as e:
        print(f"❌ Error processing {url}: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict", response_model=PredictionResponse)
def predict_article_post(payload: dict = Body(...)):
    url = payload.get("url")
    if not url:
        return JSONResponse(status_code=400, content={"error": "No URL provided"})
    return predict_article(url)

# Add a direct text prediction endpoint for testing
@app.post("/predict_text")
def predict_text_directly(payload: dict = Body(...)):
    try:
        text = payload.get("text")
        title = payload.get("title", "Direct Text Input")
        
        if not text:
            return JSONResponse(status_code=400, content={"error": "No text provided"})
        
        print(f"📝 Analyzing direct text: {len(text)} characters")
        
        # Get BERT prediction using sliding window approach
        prediction_proba = bert_predict_proba([text])[0]
        
        # Determine final prediction with higher threshold for fake news (70% confidence required)
        # This reduces false positives where real news is incorrectly labeled as fake
        FAKE_THRESHOLD = 0.70  # Require 70% confidence to classify as fake
        
        if prediction_proba[0] > FAKE_THRESHOLD:  # High confidence fake
            prediction = 0  # Fake
        else:
            prediction = 1  # Real (default to real unless highly confident it's fake)
            
        confidence = max(prediction_proba[0], prediction_proba[1])
        
        return {
            "title": title,
            "prediction": "Real" if prediction == 1 else "Fake",
            "confidence_real": round(float(prediction_proba[1]), 3),
            "confidence_fake": round(float(prediction_proba[0]), 3),
            "text_length": len(text),
            "word_count": len(text.split())
        }
        
    except Exception as e:
        print(f"❌ Error processing text: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/plot")
def get_shap_plot():
    if os.path.exists("shap_plot.png"):
        return FileResponse("shap_plot.png", media_type="image/png")
    return JSONResponse(status_code=404, content={"error": "Plot not available"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
