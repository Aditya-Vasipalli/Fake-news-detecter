import pandas as pd
import numpy as np
import shap
import random
import matplotlib.pyplot as plt
from newspaper import Article  # Import the Article class from newspaper3k
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load Dataset
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")

real['label'] = 0  # Real
fake['label'] = 1  # Fake

data = pd.concat([real, fake], axis=0).sample(frac=1).reset_index(drop=True)
data['content'] = data['title'] + " " + data['text']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['content'], data['label'], test_size=0.2, random_state=42
)

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save the trained model
joblib.dump(model, "model.pkl")

# Save the vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# Evaluate
y_pred = model.predict(X_test_vec)
print("\n=== Classification Report ===\n")
print(classification_report(y_test, y_pred))

# Pick a random article from test set
sample_idx = random.randint(0, X_test_vec.shape[0] - 1)
sample_text = X_test.iloc[sample_idx]
true_label = y_test.iloc[sample_idx]

print(f"\n🔍 Sample Text:\n{sample_text[:500]}...\n")
print(f"True Label: {'Fake' if true_label == 1 else 'Real'}")

# SHAP Explainability
explainer = shap.Explainer(model, X_train_vec)
sample_text_vectorized = vectorizer.transform([sample_text])
shap_values = explainer(sample_text_vectorized)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
sample_values = shap_values[0].values
sample_indices = sample_text_vectorized.nonzero()[1]  # Get non-zero indices from the sparse vector

# Get top features and their SHAP values
top_n = 10
non_zero_values = sample_values[sample_indices]  # SHAP values for non-zero features
top_idx = np.argsort(np.abs(non_zero_values))[-top_n:][::-1]  # Top N influential words
top_keywords = [(feature_names[sample_indices[i]], non_zero_values[i]) for i in top_idx]

print("\n=== Top Influential Words (SHAP) ===")
for word, val in top_keywords:
    direction = "Fake ⬆️" if val > 0 else "Real ⬇️"
    print(f"{word:<15} | {val:+.3f} | {direction}")

# Plot the SHAP values
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[0], show=False)
plt.tight_layout()
plt.show()

# URL of the article to analyze
url = 'https://www.bbc.com/news/articles/c2kv93y4289o'

# Fetch and parse the article
article = Article(url)
article.download()
article.parse()

# Combine the title and text
text = article.title + " " + article.text

# Preprocess the article text
article_vectorized = vectorizer.transform([text])  # Transform the text using the trained vectorizer

# Predict using the trained model
prediction = model.predict(article_vectorized)[0]  # Get the prediction (0 for Real, 1 for Fake)
prediction_proba = model.predict_proba(article_vectorized)[0]  # Get prediction probabilities

# Display the results
print(f"\n🔍 Article Title: {article.title}")
print(f"\n📝 Article Text (First 500 characters):\n{text[:500]}...\n")
print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: Real: {prediction_proba[0]:.2f}, Fake: {prediction_proba[1]:.2f}")

# SHAP Explainability for the article
shap_values = explainer(article_vectorized)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
article_values = shap_values[0].values
article_indices = article_vectorized.nonzero()[1]  # Get non-zero indices from the sparse vector

# Get top features and their SHAP values
top_n = 10
non_zero_values = article_values[article_indices]  # SHAP values for non-zero features
top_idx = np.argsort(np.abs(non_zero_values))[-top_n:][::-1]  # Top N influential words
top_keywords = [(feature_names[article_indices[i]], non_zero_values[i]) for i in top_idx]

print("\n=== Top Influential Words (SHAP) ===")
for word, val in top_keywords:
    direction = "Fake ⬆️" if val > 0 else "Real ⬇️"
    print(f"{word:<15} | {val:+.3f} | {direction}")

# Plot the SHAP values
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[0], show=False)
plt.tight_layout()
plt.show()
