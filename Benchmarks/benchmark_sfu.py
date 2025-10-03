"""
SFU Benchmark Script - Emergent, PolitiFact, and Snopes Datasets
Benchmarks the fake news detection model on Simon Fraser University datasets
Using GPU optimization pattern from FEVER benchmark
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SFUDataset(Dataset):
    """Dataset class for SFU benchmark data with GPU optimization"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with padding and truncation
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def load_emergent_data(file_path):
    """Load and process Emergent dataset"""
    print(f"Loading Emergent dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Use article text from phase 2
    texts = df['original_article_text_phase2'].fillna('').tolist()
    
    # Map fact tags to binary labels
    # true/false -> 1/0, unverified -> skip or treat as false
    label_mapping = {
        'true': 1,
        'false': 0,
        'unverified': 0  # Treat unverified as fake for binary classification
    }
    
    labels = []
    filtered_texts = []
    
    for i, row in df.iterrows():
        fact_tag = str(row['fact_tag_phase1']).lower().strip()
        if fact_tag in label_mapping and pd.notna(row['original_article_text_phase2']):
            labels.append(label_mapping[fact_tag])
            filtered_texts.append(texts[i])
    
    print(f"Emergent: {len(filtered_texts)} samples loaded")
    return filtered_texts, labels

def load_politifact_data(file_path):
    """Load and process PolitiFact dataset"""
    print(f"Loading PolitiFact dataset from {file_path}...")
    # Handle CSV parsing issues with error handling
    try:
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"Error reading CSV, trying with different parameters: {e}")
        df = pd.read_csv(file_path, on_bad_lines='skip', quoting=1, low_memory=False)
    
    # Use article text from phase 2
    texts = df['original_article_text_phase2'].fillna('').tolist()
    
    # Map PolitiFact ratings to binary labels
    # True ratings: "True", "Mostly True", "Half-True" -> 1
    # False ratings: "Mostly False", "False", "Pants on Fire!" -> 0
    true_ratings = ['true', 'mostly true', 'half-true', 'half true']
    false_ratings = ['mostly false', 'false', 'pants on fire!', 'pants on fire']
    
    labels = []
    filtered_texts = []
    
    for i, row in df.iterrows():
        fact_tag = str(row['fact_tag_phase1']).lower().strip()
        if pd.notna(row['original_article_text_phase2']) and row['original_article_text_phase2'].strip():
            if fact_tag in true_ratings:
                labels.append(1)
                filtered_texts.append(texts[i])
            elif fact_tag in false_ratings:
                labels.append(0)
                filtered_texts.append(texts[i])
    
    print(f"PolitiFact: {len(filtered_texts)} samples loaded")
    return filtered_texts, labels

def load_snopes_data(file_path):
    """Load and process Snopes dataset"""
    print(f"Loading Snopes dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Use article text from phase 2
    texts = df['original_article_text_phase2'].fillna('').tolist()
    
    # Map Snopes ratings to binary labels
    # The fact_rating_phase1 column contains boolean values
    labels = []
    filtered_texts = []
    
    for i, row in df.iterrows():
        fact_rating = str(row['fact_rating_phase1']).lower().strip()
        if pd.notna(row['original_article_text_phase2']) and row['original_article_text_phase2'].strip():
            if fact_rating == 'true':
                labels.append(1)
                filtered_texts.append(texts[i])
            elif fact_rating == 'false':
                labels.append(0)
                filtered_texts.append(texts[i])
    
    print(f"Snopes: {len(filtered_texts)} samples loaded")
    return filtered_texts, labels

def predict_batch(model, dataloader, device):
    """Make predictions on batches with GPU optimization"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    print("Making predictions with GPU acceleration...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Move back to CPU and store
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {(i + 1) * dataloader.batch_size} samples...")
    
    return np.array(all_predictions), np.array(all_labels)

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist()
    }

def benchmark_sfu_datasets():
    """Main benchmarking function with GPU optimization"""
    print("🚀 Starting SFU Datasets Benchmark with GPU Optimization")
    print("=" * 60)
    
    # GPU Setup - Same pattern as FEVER benchmark
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model_path = "fake_news_bert_model.pth"
    tokenizer = BertTokenizer.from_pretrained('bert_tokenizer')
    
    # Initialize model
    model = BertForSequenceClassification.from_pretrained(
        'bert_model',
        num_labels=2
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("✅ Model loaded successfully")
    
    # Dataset paths
    base_path = "Benchmark_data"
    datasets = {
        'emergent': os.path.join(base_path, 'emergent_phase2_clean_2018_7_2.csv'),
        'politifact': os.path.join(base_path, 'politifact_phase2_clean_2018_7_3.csv'),
        'snopes': os.path.join(base_path, 'snopes_phase2_clean_2018_7_3.csv')
    }
    
    results = {}
    
    # Benchmark each dataset
    for dataset_name, file_path in datasets.items():
        print(f"\n{'='*20} {dataset_name.upper()} BENCHMARK {'='*20}")
        
        try:
            # Load dataset
            if dataset_name == 'emergent':
                texts, labels = load_emergent_data(file_path)
            elif dataset_name == 'politifact':
                texts, labels = load_politifact_data(file_path)
            elif dataset_name == 'snopes':
                texts, labels = load_snopes_data(file_path)
            
            if len(texts) == 0:
                print(f"❌ No valid samples found in {dataset_name}")
                continue
            
            # Create dataset and dataloader
            dataset = SFUDataset(texts, labels, tokenizer)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            print(f"Dataset: {len(texts)} samples")
            print(f"Label distribution - Real: {labels.count(1)}, Fake: {labels.count(0)}")
            
            # Make predictions
            predictions, true_labels = predict_batch(model, dataloader, device)
            
            # Calculate metrics
            metrics = calculate_metrics(true_labels, predictions)
            
            # Store results
            results[dataset_name] = {
                'dataset_size': len(texts),
                'label_distribution': {'real': labels.count(1), 'fake': labels.count(0)},
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print results
            print(f"\n📊 {dataset_name.upper()} RESULTS:")
            print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
            print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
            
            # Per-class results
            print(f"\nPer-class results:")
            print(f"Fake News - Precision: {metrics['precision_per_class'][0]:.4f}, Recall: {metrics['recall_per_class'][0]:.4f}, F1: {metrics['f1_per_class'][0]:.4f}")
            print(f"Real News - Precision: {metrics['precision_per_class'][1]:.4f}, Recall: {metrics['recall_per_class'][1]:.4f}, F1: {metrics['f1_per_class'][1]:.4f}")
            
            # Confusion Matrix
            cm = metrics['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"              Predicted")
            print(f"           Fake    Real")
            print(f"Actual Fake {cm[0][0]:4d}    {cm[0][1]:4d}")
            print(f"       Real {cm[1][0]:4d}    {cm[1][1]:4d}")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
            results[dataset_name] = {'error': str(e)}
    
    # Save comprehensive results
    output_file = 'sfu_benchmark_results.json'
    results['benchmark_info'] = {
        'model_path': model_path,
        'tokenizer_path': 'bert_tokenizer',
        'device': str(device),
        'batch_size': 16,
        'max_length': 512,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print("\n🎉 SFU Benchmark Complete!")
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    for dataset_name in ['emergent', 'politifact', 'snopes']:
        if dataset_name in results and 'metrics' in results[dataset_name]:
            acc = results[dataset_name]['metrics']['accuracy']
            size = results[dataset_name]['dataset_size']
            print(f"{dataset_name.capitalize():10}: {acc:.4f} ({acc*100:.2f}%) - {size:,} samples")

if __name__ == "__main__":
    benchmark_sfu_datasets()