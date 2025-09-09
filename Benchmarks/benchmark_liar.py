"""
LIAR Dataset Benchmark for BERT Fake News Detection Model
========================================================

This script benchmarks our trained BERT model against the LIAR dataset,
which contains real political statements with fact-checking labels.

LIAR Dataset Labels:
- true: Completely accurate
- mostly-true: Mostly accurate  
- half-true: Half accurate
- barely-true: Mostly inaccurate
- false: Completely false
- pants-fire: Ridiculously false

For our binary classification:
- REAL: true, mostly-true
- FAKE: false, pants-fire, barely-true, half-true
"""
#find dataset at https://github.com/tfs4/liar_dataset
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from tqdm.auto import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

class LIARBenchmark:
    def __init__(self, model_path='./bert_model', tokenizer_path='./bert_tokenizer'):
        """Initialize the benchmark with trained BERT model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
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
        
        # Label mapping for LIAR dataset
        self.label_mapping = {
            'true': 1,           # REAL
            'mostly-true': 1,    # REAL  
            'half-true': 0,      # FAKE (leaning false)
            'barely-true': 0,    # FAKE
            'false': 0,          # FAKE
            'pants-fire': 0      # FAKE
        }
        
    def load_liar_data(self, file_path):
        """Load and process LIAR dataset TSV file"""
        print(f"📁 Loading LIAR data from: {file_path}")
        
        try:
            # TSV format: ID, label, statement, subject, speaker, job, state, party, counts..., context
            df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')
            
            # Extract relevant columns
            df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 
                         'state', 'party', 'barely_true_count', 'false_count', 
                         'half_true_count', 'mostly_true_count', 'pants_fire_count', 'context']
            
            # Convert labels to binary
            df['binary_label'] = df['label'].map(self.label_mapping)
            
            # Remove rows with unknown labels
            df = df.dropna(subset=['binary_label'])
            
            # Clean statements
            df['statement'] = df['statement'].astype(str).str.strip()
            df = df[df['statement'].str.len() > 10]  # Remove very short statements
            
            print(f"✅ Loaded {len(df)} samples")
            print(f"📊 Label distribution:")
            label_counts = df['label'].value_counts()
            for label, count in label_counts.items():
                binary = "REAL" if self.label_mapping.get(label, 0) == 1 else "FAKE"
                print(f"   {label}: {count} ({binary})")
            
            binary_counts = df['binary_label'].value_counts()
            print(f"📈 Binary distribution:")
            print(f"   REAL (1): {binary_counts.get(1, 0)}")
            print(f"   FAKE (0): {binary_counts.get(0, 0)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def predict_text(self, text, max_length=512):
        """Predict single text using BERT model"""
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probabilities, dim=-1)
            
        return prediction.cpu().numpy()[0], probabilities.cpu().numpy()[0]
    
    def predict_batch(self, texts, batch_size=16, max_length=512):
        """Predict batch of texts efficiently"""
        predictions = []
        probabilities = []
        
        print(f"🔮 Predicting {len(texts)} samples in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_probabilities = torch.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(batch_probabilities, dim=-1)
                
                predictions.extend(batch_predictions.cpu().numpy())
                probabilities.extend(batch_probabilities.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)
    
    def benchmark_dataset(self, dataset_name, file_path):
        """Benchmark model on a specific dataset split"""
        print(f"\n🎯 Benchmarking on {dataset_name.upper()} set")
        print("=" * 50)
        
        # Load data
        df = self.load_liar_data(file_path)
        if df is None:
            return None
        
        # Get predictions
        texts = df['statement'].tolist()
        true_labels = df['binary_label'].tolist()
        
        predictions, probabilities = self.predict_batch(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, predictions, 
            target_names=['FAKE', 'REAL'],
            digits=4
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Results summary
        results = {
            'dataset': dataset_name,
            'total_samples': len(df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'original_labels': df['label'].tolist(),
            'statements': texts
        }
        
        # Print results
        print(f"📊 Results for {dataset_name.upper()}:")
        print(f"   Total samples: {len(df)}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"\n📈 Detailed Classification Report:")
        print(class_report)
        print(f"\n🎭 Confusion Matrix:")
        print(f"   True\\Pred  FAKE  REAL")
        print(f"   FAKE      {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"   REAL      {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return results
    
    def analyze_errors(self, results):
        """Analyze prediction errors in detail"""
        print(f"\n🔍 Error Analysis for {results['dataset'].upper()}")
        print("=" * 50)
        
        predictions = results['predictions']
        true_labels = results['true_labels']
        original_labels = results['original_labels']
        statements = results['statements']
        probabilities = results['probabilities']
        
        # Find errors
        errors = []
        for i in range(len(predictions)):
            if predictions[i] != true_labels[i]:
                errors.append({
                    'index': i,
                    'statement': statements[i][:200] + "...",
                    'original_label': original_labels[i],
                    'true_binary': "REAL" if true_labels[i] == 1 else "FAKE",
                    'predicted': "REAL" if predictions[i] == 1 else "FAKE",
                    'confidence': max(probabilities[i]),
                    'prob_fake': probabilities[i][0],
                    'prob_real': probabilities[i][1]
                })
        
        print(f"❌ Total errors: {len(errors)} out of {len(predictions)} ({len(errors)/len(predictions)*100:.2f}%)")
        
        # Analyze error patterns by original label
        error_by_original = {}
        for error in errors:
            label = error['original_label']
            error_by_original[label] = error_by_original.get(label, 0) + 1
        
        print(f"\n📊 Errors by original LIAR label:")
        for label, count in sorted(error_by_original.items()):
            binary = "REAL" if self.label_mapping.get(label, 0) == 1 else "FAKE"
            print(f"   {label} ({binary}): {count} errors")
        
        # Show most confident wrong predictions
        confident_errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)[:10]
        
        print(f"\n🎯 Top 10 Most Confident Wrong Predictions:")
        for i, error in enumerate(confident_errors, 1):
            print(f"\n{i}. Original: {error['original_label']} → True: {error['true_binary']} → Predicted: {error['predicted']}")
            print(f"   Confidence: {error['confidence']:.4f} (Fake: {error['prob_fake']:.3f}, Real: {error['prob_real']:.3f})")
            print(f"   Statement: {error['statement']}")
        
        return errors
    
    def run_full_benchmark(self):
        """Run complete benchmark on all LIAR dataset splits"""
        print("🚀 Starting Full LIAR Dataset Benchmark")
        print("=" * 60)
        
        datasets = [
            ('train', './liar_dataset-master/train.tsv'),
            ('validation', './liar_dataset-master/valid.tsv'),
            ('test', './liar_dataset-master/test.tsv')
        ]
        
        all_results = {}
        
        for dataset_name, file_path in datasets:
            if os.path.exists(file_path):
                results = self.benchmark_dataset(dataset_name, file_path)
                if results:
                    all_results[dataset_name] = results
                    # Analyze errors for this dataset
                    self.analyze_errors(results)
            else:
                print(f"⚠️ File not found: {file_path}")
        
        # Overall summary
        print(f"\n🎉 OVERALL BENCHMARK SUMMARY")
        print("=" * 60)
        
        for dataset_name, results in all_results.items():
            print(f"{dataset_name.upper():12} | "
                  f"Samples: {results['total_samples']:5d} | "
                  f"Accuracy: {results['accuracy']:.4f} | "
                  f"F1: {results['f1_score']:.4f}")
        
        # Create visualization
        self.create_benchmark_plots(all_results)
        
        return all_results
    
    def create_benchmark_plots(self, all_results):
        """Create visualization plots for benchmark results"""
        print(f"\n📊 Attempting to create benchmark visualization plots...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns # type: ignore
            
            # Set up the plot style
            plt.style.use('default')
            sns.set_palette("husl")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('BERT Model Performance on LIAR Dataset', fontsize=16, fontweight='bold')
            
            # 1. Accuracy comparison
            datasets = list(all_results.keys())
            accuracies = [all_results[d]['accuracy'] for d in datasets]
            
            axes[0,0].bar(datasets, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0,0].set_title('Accuracy by Dataset Split')
            axes[0,0].set_ylabel('Accuracy')
            axes[0,0].set_ylim(0, 1)
            for i, acc in enumerate(accuracies):
                axes[0,0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
            
            # 2. F1 Score comparison
            f1_scores = [all_results[d]['f1_score'] for d in datasets]
            
            axes[0,1].bar(datasets, f1_scores, color=['#FF9F43', '#10C469', '#5B73E8'])
            axes[0,1].set_title('F1-Score by Dataset Split')
            axes[0,1].set_ylabel('F1-Score')
            axes[0,1].set_ylim(0, 1)
            for i, f1 in enumerate(f1_scores):
                axes[0,1].text(i, f1 + 0.01, f'{f1:.3f}', ha='center', fontweight='bold')
            
            # 3. Confusion Matrix for Test Set (if available)
            if 'test' in all_results:
                cm = all_results['test']['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['FAKE', 'REAL'], 
                           yticklabels=['FAKE', 'REAL'],
                           ax=axes[1,0])
                axes[1,0].set_title('Confusion Matrix (Test Set)')
                axes[1,0].set_ylabel('True Label')
                axes[1,0].set_xlabel('Predicted Label')
            
            # 4. Precision-Recall comparison
            precisions = [all_results[d]['precision'] for d in datasets]
            recalls = [all_results[d]['recall'] for d in datasets]
            
            x = np.arange(len(datasets))
            width = 0.35
            
            axes[1,1].bar(x - width/2, precisions, width, label='Precision', color='#E74C3C')
            axes[1,1].bar(x + width/2, recalls, width, label='Recall', color='#3498DB')
            axes[1,1].set_title('Precision vs Recall by Dataset')
            axes[1,1].set_ylabel('Score')
            axes[1,1].set_xlabel('Dataset')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(datasets)
            axes[1,1].legend()
            axes[1,1].set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig('liar_benchmark_results.png', dpi=300, bbox_inches='tight')
            print("✅ Saved benchmark plots to: liar_benchmark_results.png")
            
        except ImportError:
            print("⚠️ Matplotlib/Seaborn not available - skipping plots")
        except Exception as e:
            print(f"⚠️ Could not create plots: {e}")
    
    def save_results(self, all_results, filename='liar_benchmark_results.json'):
        """Save benchmark results to JSON file"""
        print(f"💾 Saving results to {filename}...")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset, results in all_results.items():
            json_results[dataset] = {
                'dataset': results['dataset'],
                'total_samples': results['total_samples'],
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report']
            }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {filename}")

def main():
    """Main function to run the benchmark"""
    print("🎯 LIAR Dataset Benchmark for BERT Fake News Detection")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = LIARBenchmark()
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    # Save results
    if results:
        benchmark.save_results(results)
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"📊 Check 'liar_benchmark_results.png' for visualizations")
    print(f"💾 Check 'liar_benchmark_results.json' for detailed results")

if __name__ == "__main__":
    main()

"""
OVERALL BENCHMARK SUMMARY
============================================================
TRAIN        | Samples: 10240 | Accuracy: 0.6415 | F1: 0.5127
VALIDATION   | Samples:  1284 | Accuracy: 0.6690 | F1: 0.5490
TEST         | Samples:  1267 | Accuracy: 0.6401 | F1: 0.5095

"""