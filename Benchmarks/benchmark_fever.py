"""
FEVER Dataset Benchmark for BERT Fake News Detection Model
==========================================================

This script evaluates our trained BERT model on the FEVER (Fact Extraction and VERification) dataset.
FEVER contain        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probabilities = torch.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(batch_probabilities, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())
                
                # Clear GPU cache periodically to prevent memory issues
                if torch.cuda.is_available() and (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {(i + 1) * batch_size} samples...")
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1e9
                        print(f"   GPU Memory Used: {memory_used:.2f} GB")need to be verified against Wikipedia evidence.

Dataset Structure:
- Claims with evidence from Wikipedia
- Labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
- Mapping: SUPPORTS → REAL, REFUTES/NOT ENOUGH INFO → FAKE

Authors: Aditya Vasipalli, Aditya Kachwaha
Date: October 3, 2025
"""

import json
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import BERT model utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FEVERDataset(Dataset):
    """Dataset class for FEVER data"""
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
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class FEVERBenchmark:
    """FEVER Dataset Benchmark Class"""
    
    def __init__(self, model_path="./bert_model", tokenizer_path="./bert_tokenizer"):
        """Initialize the benchmark with trained BERT model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"� Using device: {self.device}")
        
        # Load model and tokenizer
        print("📚 Loading BERT model and tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, 
            num_labels=2,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def load_fever_data(self, fever_folder="./FEVER"):
        """Load and process FEVER dataset"""
        print(f"\n📂 Loading FEVER data from {fever_folder}...")
        
        texts = []
        labels = []
        
        # Look for FEVER JSON files
        fever_files = []
        if os.path.exists(fever_folder):
            for file in os.listdir(fever_folder):
                if file.endswith('.jsonl') or file.endswith('.json'):
                    fever_files.append(os.path.join(fever_folder, file))
        
        if not fever_files:
            print(f"❌ No FEVER dataset files found in {fever_folder}")
            return [], []
        
        print(f"📄 Found {len(fever_files)} FEVER files:")
        for file in fever_files:
            print(f"   - {os.path.basename(file)}")
        
        # Process each file
        total_processed = 0
        for file_path in fever_files:
            try:
                print(f"\n🔍 Processing {os.path.basename(file_path)}...")
                file_texts, file_labels = self._process_fever_file(file_path)
                texts.extend(file_texts)
                labels.extend(file_labels)
                total_processed += len(file_texts)
                print(f"   ✅ Processed {len(file_texts)} samples")
            except Exception as e:
                print(f"   ❌ Error processing {file_path}: {e}")
                continue
        
        print(f"\n📊 FEVER Dataset Summary:")
        print(f"   Total samples: {len(texts)}")
        print(f"   REAL samples: {sum(1 for label in labels if label == 1)}")
        print(f"   FAKE samples: {sum(1 for label in labels if label == 0)}")
        
        return texts, labels
    
    def _process_fever_file(self, file_path):
        """Process individual FEVER file"""
        texts = []
        labels = []
        
        # Determine file format
        if file_path.endswith('.jsonl'):
            # JSONL format (one JSON object per line)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        text, label = self._extract_fever_sample(data)
                        if text and label is not None:
                            texts.append(text)
                            labels.append(label)
                    except json.JSONDecodeError as e:
                        if line_num < 10:  # Only log first few errors
                            print(f"     ⚠️ JSON decode error on line {line_num + 1}: {e}")
                        continue
                    except Exception as e:
                        if line_num < 10:
                            print(f"     ⚠️ Processing error on line {line_num + 1}: {e}")
                        continue
        else:
            # Regular JSON format
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        text, label = self._extract_fever_sample(item)
                        if text and label is not None:
                            texts.append(text)
                            labels.append(label)
                elif isinstance(data, dict):
                    text, label = self._extract_fever_sample(data)
                    if text and label is not None:
                        texts.append(text)
                        labels.append(label)
            except json.JSONDecodeError as e:
                print(f"     ❌ Invalid JSON file: {e}")
            except Exception as e:
                print(f"     ❌ Error reading file: {e}")
        
        return texts, labels
    
    def _extract_fever_sample(self, data):
        """Extract text and label from FEVER data sample"""
        try:
            # Common FEVER fields
            claim = data.get('claim', '')
            label = data.get('label', data.get('verifiable', ''))
            evidence = data.get('evidence', [])
            
            # Build text from claim and evidence
            text_parts = []
            
            # Add claim
            if claim:
                text_parts.append(f"Claim: {claim}")
            
            # Add evidence if available
            if evidence and isinstance(evidence, list):
                evidence_texts = []
                for ev in evidence[:3]:  # Limit to first 3 evidence pieces
                    if isinstance(ev, dict):
                        # Handle different evidence formats
                        ev_text = ev.get('text', ev.get('evidence', ''))
                        if ev_text:
                            evidence_texts.append(str(ev_text))
                    elif isinstance(ev, str):
                        evidence_texts.append(ev)
                
                if evidence_texts:
                    text_parts.append(f"Evidence: {' '.join(evidence_texts)}")
            
            # Combine text
            text = ' '.join(text_parts).strip()
            
            # Map labels to binary classification
            label_mapping = {
                'SUPPORTS': 1,      # REAL
                'REFUTES': 0,       # FAKE
                'NOT ENOUGH INFO': 0,  # FAKE (cannot verify)
                'VERIFIABLE': 1,    # REAL
                'NOT VERIFIABLE': 0  # FAKE
            }
            
            # Normalize label
            if isinstance(label, str):
                label = label.upper().strip()
                binary_label = label_mapping.get(label)
            else:
                binary_label = None
            
            # Return only if we have both text and valid label
            if text and binary_label is not None:
                return text, binary_label
            
            return None, None
            
        except Exception as e:
            return None, None
    
    def predict_batch(self, texts, batch_size=16):
        """Predict labels for a batch of texts"""
        # Create dummy labels for dataset (not used in evaluation)
        dummy_labels = [0] * len(texts)
        dataset = FEVERDataset(texts, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        probabilities = []
        
        print(f"🔮 Running inference on {len(texts)} samples...")
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probabilities = torch.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(batch_probabilities, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {(i + 1) * batch_size} samples...")
        
        return predictions, probabilities
    
    def evaluate_fever(self):
        """Evaluate model on FEVER dataset"""
        print("🎯 Starting FEVER Dataset Evaluation")
        print("=" * 50)
        
        start_time = time.time()
        
        # Load FEVER data
        texts, labels = self.load_fever_data()
        
        if not texts:
            print("❌ No data loaded. Exiting...")
            return None
        
        # Run predictions
        predictions, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Detailed metrics per class
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Processing time
        processing_time = time.time() - start_time
        
        # Print results
        print(f"\n🎯 FEVER Dataset Evaluation Results")
        print("=" * 50)
        print(f"📊 Dataset Size: {len(texts)} samples")
        print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
        print(f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 Weighted Precision: {precision:.4f}")
        print(f"📈 Weighted Recall: {recall:.4f}")
        print(f"📈 Weighted F1-Score: {f1:.4f}")
        
        print(f"\n📋 Per-Class Metrics:")
        class_names = ['FAKE', 'REAL']
        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                print(f"   {class_name:>4}: Precision={precision_per_class[i]:.4f}, "
                      f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}, "
                      f"Support={support_per_class[i]}")
        
        print(f"\n🔍 Confusion Matrix:")
        print(f"                Predicted")
        print(f"                FAKE  REAL")
        print(f"Actual   FAKE   {cm[0][0]:>4}  {cm[0][1]:>4}")
        print(f"         REAL   {cm[1][0]:>4}  {cm[1][1]:>4}")
        
        # Classification report
        print(f"\n📊 Detailed Classification Report:")
        print(classification_report(labels, predictions, target_names=class_names, zero_division=0))
        
        # Show some examples
        print(f"\n📝 Sample Predictions:")
        for i in range(min(5, len(texts))):
            actual = "REAL" if labels[i] == 1 else "FAKE"
            predicted = "REAL" if predictions[i] == 1 else "FAKE"
            confidence = max(probabilities[i])
            
            print(f"\n   Sample {i+1}:")
            print(f"   Text: {texts[i][:200]}..." if len(texts[i]) > 200 else f"   Text: {texts[i]}")
            print(f"   Actual: {actual}, Predicted: {predicted}, Confidence: {confidence:.4f}")
        
        # Prepare results for export
        results = {
            'dataset': 'FEVER',
            'evaluation_date': datetime.now().isoformat(),
            'dataset_size': len(texts),
            'processing_time_seconds': processing_time,
            'metrics': {
                'accuracy': float(accuracy),
                'weighted_precision': float(precision),
                'weighted_recall': float(recall),
                'weighted_f1': float(f1)
            },
            'per_class_metrics': {
                'fake': {
                    'precision': float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
                    'recall': float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
                    'f1': float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
                    'support': int(support_per_class[0]) if len(support_per_class) > 0 else 0
                },
                'real': {
                    'precision': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
                    'recall': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
                    'f1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                    'support': int(support_per_class[1]) if len(support_per_class) > 1 else 0
                }
            },
            'confusion_matrix': cm.tolist(),
            'label_distribution': {
                'fake': int(sum(1 for label in labels if label == 0)),
                'real': int(sum(1 for label in labels if label == 1))
            }
        }
        
        # Save results
        results_file = 'fever_benchmark_results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {results_file}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")
        
        print(f"\n✅ FEVER evaluation completed!")
        return results

def main():
    """Main evaluation function"""
    print("🔥 FEVER Dataset Benchmark for BERT Fake News Detection")
    print("=" * 60)
    
    try:
        # Initialize benchmark
        benchmark = FEVERBenchmark()
        
        # Run evaluation
        results = benchmark.evaluate_fever()
        
        if results:
            print(f"\n🎉 FEVER benchmark completed successfully!")
            print(f"📊 Overall Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"📈 F1-Score: {results['metrics']['weighted_f1']:.4f}")
        else:
            print("❌ Benchmark failed.")
            
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()