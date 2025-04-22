import pandas as pd
import numpy as np
import shap
import random
import matplotlib.pyplot as plt
from newspaper import Article
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

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

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the data
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

train_dataset = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
test_dataset = NewsDataset(X_test.tolist(), y_test.tolist(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluation
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

print("\n=== Classification Report ===\n")
print(classification_report(y_true, y_pred))

# Predicting a real article
url = 'https://www.bbc.com/news/live/crknlnzlrzdt'
article = Article(url)
article.download()
article.parse()

text = article.title + " " + article.text
encoding = tokenizer(
    text,
    max_length=512,
    padding='max_length',
    truncation=True,
    return_tensors="pt"
)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    prediction_proba = torch.softmax(logits, dim=1).cpu().numpy()

print(f"\n🔍 Article Title: {article.title}")
print(f"\n📝 Article Text (First 500 characters):\n{text[:500]}...\n")
print(f"Prediction: {'Fake' if prediction == 1 else 'Real'}")
print(f"Confidence: Real: {prediction_proba[0][0]:.2f}, Fake: {prediction_proba[0][1]:.2f}")
