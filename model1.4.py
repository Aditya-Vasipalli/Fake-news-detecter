import pandas as pd
import numpy as np
import shap
import random
import matplotlib.pyplot as plt

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
explainer = shap.Explainer(model, X_train_vec)  # Use the training data to initialize the explainer
sample_text_vectorized = vectorizer.transform([sample_text])  # Vectorize the sample text
shap_values = explainer(sample_text_vectorized)

print("\n=== Top Features (SHAP) ===")
# Plot the SHAP values for the sample text
shap.plots.bar(shap_values[0])  # Use bar plot for sparse data
