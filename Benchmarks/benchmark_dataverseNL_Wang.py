"""
DataverseNL Kevin Wang Dataset Benchmark for BERT Fake News Detection Model
===========================================================================

This script benchmarks our trained BERT model against multiple datasets from the
DataverseNL Kevin Wang collection, which includes various fake news datasets like
FA-KES, WELFake, Kaggle datasets, and more.

Dataset Collection:
- FA-KES-Dataset.csv: FA-KES fake news dataset
- WELFake_Dataset.csv: WEL and Fake news dataset
- Kaggle1.csv & Kaggle2.csv: Kaggle fake news datasets
- merged.csv: Combined dataset
- SCRAPED.csv: Scraped news data
- Various trimmed versions
"""

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
import glob
warnings.filterwarnings('ignore')

class DataverseNLBenchmark:
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
        
        # Dataset paths
        self.dataset_folder = "C:/Users/vasip/Code/Fake-news-detecter/dataverseNL/Kevin Wang/WEBAPP/datasets"
        
    def detect_dataset_format(self, df, filename):
        """Detect the format and column structure of each dataset"""
        print(f"🔍 Analyzing dataset format for: {filename}")
        
        columns = df.columns.tolist()
        print(f"   Columns: {columns}")
        print(f"   Shape: {df.shape}")
        
        # Try to identify text and label columns
        text_column = None
        label_column = None
        
        # Common text column names
        text_candidates = ['text', 'article_content', 'content', 'news', 'statement', 'title']
        for col in columns:
            if any(candidate in col.lower() for candidate in text_candidates):
                text_column = col
                break
        
        # If no direct match, look for longest text column
        if text_column is None:
            text_lengths = {}
            for col in columns:
                if df[col].dtype == 'object':  # String column
                    try:
                        avg_length = df[col].astype(str).str.len().mean()
                        text_lengths[col] = avg_length
                    except:
                        continue
            if text_lengths:
                text_column = max(text_lengths, key=text_lengths.get)
        
        # Common label column names
        label_candidates = ['label', 'labels', 'fake', 'class', 'target', 'y']
        for col in columns:
            if col.lower() in label_candidates:
                label_column = col
                break
        
        print(f"   Text column: {text_column}")
        print(f"   Label column: {label_column}")
        
        if text_column and label_column:
            # Analyze label distribution
            unique_labels = df[label_column].unique()
            print(f"   Unique labels: {unique_labels}")
            
            label_counts = df[label_column].value_counts()
            print(f"   Label distribution: {dict(label_counts)}")
        
        return text_column, label_column
    
    def standardize_labels(self, df, label_column, filename):
        """Convert various label formats to binary (0=FAKE, 1=REAL)"""
        unique_labels = df[label_column].unique()
        print(f"📊 Standardizing labels for {filename}: {unique_labels}")
        
        # Create label mapping based on dataset patterns
        label_mapping = {}
        
        for label in unique_labels:
            label_str = str(label).lower().strip()
            
            # Binary numeric labels
            if label in [0, '0', 0.0]:
                label_mapping[label] = 0  # FAKE
            elif label in [1, '1', 1.0]:
                label_mapping[label] = 1  # REAL
            
            # String labels
            elif 'fake' in label_str or 'false' in label_str:
                label_mapping[label] = 0  # FAKE
            elif 'real' in label_str or 'true' in label_str or 'reliable' in label_str:
                label_mapping[label] = 1  # REAL
            
            # Default: assume first unique value is 0, second is 1
            else:
                sorted_labels = sorted(unique_labels)
                if label == sorted_labels[0]:
                    label_mapping[label] = 0
                else:
                    label_mapping[label] = 1
        
        print(f"   Label mapping: {label_mapping}")
        
        # Apply mapping
        df['binary_label'] = df[label_column].map(label_mapping)
        
        # Remove unmapped labels
        before_count = len(df)
        df = df.dropna(subset=['binary_label'])
        after_count = len(df)
        
        if before_count > after_count:
            print(f"   ⚠️ Removed {before_count - after_count} samples with unmapped labels")
        
        return df, label_mapping
    
    def load_and_process_dataset(self, filepath):
        """Load and process a single dataset file"""
        filename = os.path.basename(filepath)
        print(f"\n📁 Loading dataset: {filename}")
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    print(f"   ✅ Loaded with {encoding} encoding")
                    break
                except:
                    continue
            
            if df is None:
                print(f"   ❌ Could not load {filename} with any encoding")
                return None
            
            # Detect format
            text_column, label_column = self.detect_dataset_format(df, filename)
            
            if not text_column or not label_column:
                print(f"   ❌ Could not identify text and label columns for {filename}")
                return None
            
            # Standardize labels
            df, label_mapping = self.standardize_labels(df, label_column, filename)
            
            # Clean text data
            df[text_column] = df[text_column].astype(str).str.strip()
            df = df[df[text_column].str.len() > 10]  # Remove very short texts
            
            # Remove duplicates
            before_dedup = len(df)
            df = df.drop_duplicates(subset=[text_column])
            after_dedup = len(df)
            
            if before_dedup > after_dedup:
                print(f"   🔄 Removed {before_dedup - after_dedup} duplicate texts")
            
            print(f"   📊 Final dataset size: {len(df)} samples")
            
            binary_counts = df['binary_label'].value_counts()
            print(f"   📈 Binary distribution: FAKE={binary_counts.get(0, 0)}, REAL={binary_counts.get(1, 0)}")
            
            return {
                'filename': filename,
                'dataframe': df,
                'text_column': text_column,
                'label_column': label_column,
                'label_mapping': label_mapping,
                'total_samples': len(df)
            }
            
        except Exception as e:
            print(f"   ❌ Error loading {filename}: {e}")
            return None
    
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
    
    def benchmark_single_dataset(self, dataset_info):
        """Benchmark model on a single dataset"""
        filename = dataset_info['filename']
        df = dataset_info['dataframe']
        text_column = dataset_info['text_column']
        
        print(f"\n🎯 Benchmarking on: {filename}")
        print("=" * 50)
        
        # Get predictions
        texts = df[text_column].tolist()
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
            'filename': filename,
            'total_samples': len(df),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'label_mapping': dataset_info['label_mapping']
        }
        
        # Print results
        print(f"📊 Results for {filename}:")
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
        
        # Error analysis
        errors = (predictions != true_labels).sum()
        error_rate = errors / len(predictions) * 100
        print(f"\n❌ Errors: {errors} ({error_rate:.2f}%)")
        
        return results
    
    def run_full_benchmark(self):
        """Run benchmark on all available datasets"""
        print("🚀 Starting DataverseNL Kevin Wang Dataset Benchmark")
        print("=" * 70)
        
        # Find all CSV files in the dataset folder
        csv_files = glob.glob(os.path.join(self.dataset_folder, "*.csv"))
        
        print(f"📂 Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"   - {os.path.basename(csv_file)}")
        
        all_results = {}
        dataset_infos = []
        
        # Load and process all datasets
        for csv_file in csv_files:
            dataset_info = self.load_and_process_dataset(csv_file)
            if dataset_info:
                dataset_infos.append(dataset_info)
        
        print(f"\n✅ Successfully loaded {len(dataset_infos)} datasets")
        
        # Benchmark each dataset
        for dataset_info in dataset_infos:
            try:
                results = self.benchmark_single_dataset(dataset_info)
                all_results[dataset_info['filename']] = results
            except Exception as e:
                print(f"❌ Error benchmarking {dataset_info['filename']}: {e}")
                continue
        
        # Overall summary
        print(f"\n🎉 OVERALL BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Dataset':<25} | {'Samples':<8} | {'Accuracy':<8} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8}")
        print("-" * 70)
        
        total_samples = 0
        weighted_accuracy = 0
        
        for filename, results in all_results.items():
            samples = results['total_samples']
            accuracy = results['accuracy']
            f1 = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            
            print(f"{filename:<25} | {samples:<8d} | {accuracy:<8.4f} | {f1:<8.4f} | {precision:<9.4f} | {recall:<8.4f}")
            
            # Calculate weighted averages
            total_samples += samples
            weighted_accuracy += accuracy * samples
        
        if total_samples > 0:
            avg_accuracy = weighted_accuracy / total_samples
            print("-" * 70)
            print(f"{'WEIGHTED AVERAGE':<25} | {total_samples:<8d} | {avg_accuracy:<8.4f} | {'':8} | {'':9} | {'':8}")
        
        return all_results
    
    def save_results(self, all_results, filename='dataverseNL_benchmark_results.json'):
        """Save benchmark results to JSON file"""
        print(f"\n💾 Saving results to {filename}...")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for dataset, results in all_results.items():
            json_results[dataset] = {
                'filename': results['filename'],
                'total_samples': results['total_samples'],
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report'],
                'label_mapping': results['label_mapping']
            }
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {filename}")

def main():
    """Main function to run the benchmark"""
    print("🎯 DataverseNL Kevin Wang Dataset Benchmark for BERT Fake News Detection")
    print("=" * 80)
    
    # Initialize benchmark
    benchmark = DataverseNLBenchmark()
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    # Save results
    if results:
        benchmark.save_results(results)
    
    print(f"\n✅ Benchmark completed successfully!")
    print(f"💾 Check 'dataverseNL_benchmark_results.json' for detailed results")

if __name__ == "__main__":
    main()

"""
OVERALL BENCHMARK SUMMARY
======================================================================
Dataset                   | Samples  | Accuracy | F1-Score | Precision | Recall
----------------------------------------------------------------------
FA-KES-Dataset.csv        | 774      | 0.4509   | 0.3933   | 0.4527    | 0.4509
Kaggle1.csv               | 19781    | 0.4746   | 0.3896   | 0.4078    | 0.4746
Kaggle2.csv               | 2863     | 0.6266   | 0.6199   | 0.6924    | 0.6266
merged.csv                | 10469    | 0.4689   | 0.3838   | 0.4342    | 0.4689
SCRAPED.csv               | 36787    | 0.5164   | 0.4903   | 0.4783    | 0.5164
Trimmed-Kaggle1.csv       | 2901     | 0.4619   | 0.3762   | 0.4018    | 0.4619
trimmed-scraped.csv       | 2879     | 0.4463   | 0.4008   | 0.4131    | 0.4463
trimmed-WEL.csv           | 2963     | 0.4641   | 0.3760   | 0.4644    | 0.4641
WELFake_Dataset.csv       | 62313    | 0.5078   | 0.4221   | 0.4361    | 0.5078
----------------------------------------------------------------------
WEIGHTED AVERAGE          | 141730   | 0.5015   | 0.4509   | 0.4783    | 0.5015


FA-KES-Dataset.csv
Results for FA-KES-Dataset.csv:
   Total samples: 774
   Accuracy: 0.4509
   Precision: 0.4527
   Recall: 0.4509
   F1-Score: 0.3933

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4499    0.7818    0.5711       362
        REAL     0.4552    0.1602    0.2370       412

    accuracy                         0.4509       774
   macro avg     0.4525    0.4710    0.4041       774
weighted avg     0.4527    0.4509    0.3933       774


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE       283    79
   REAL       346    66

❌ Errors: 425 (54.91%)

🎯 Benchmarking on: Kaggle1.csv
==================================================
🔮 Predicting 19781 samples in batches of 16...
📊 Results for Kaggle1.csv:
   Total samples: 19781
   Accuracy: 0.4746
   Precision: 0.4078
   Recall: 0.4746
   F1-Score: 0.3896

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4997    0.8288    0.6235     10381
        REAL     0.3064    0.0835    0.1312      9400

    accuracy                         0.4746     19781
   macro avg     0.4030    0.4562    0.3774     19781
weighted avg     0.4078    0.4746    0.3896     19781


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      8604  1777
   REAL      8615   785

❌ Errors: 10392 (52.54%)

🎯 Benchmarking on: Kaggle2.csv
==================================================
🔮 Predicting 2863 samples in batches of 16...
📊 Results for Kaggle2.csv:
   Total samples: 2863
   Accuracy: 0.6266
   Precision: 0.6924
   Recall: 0.6266
   F1-Score: 0.6199

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.5329    0.8407    0.6524      1193
        REAL     0.8063    0.4737    0.5968      1670

    accuracy                         0.6266      2863
   macro avg     0.6696    0.6572    0.6246      2863
weighted avg     0.6924    0.6266    0.6199      2863


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      1003   190
   REAL       879   791

❌ Errors: 1069 (37.34%)

🎯 Benchmarking on: merged.csv
==================================================
🔮 Predicting 10469 samples in batches of 16...
📊 Results for merged.csv:
   Total samples: 10469
   Accuracy: 0.4689
   Precision: 0.4342
   Recall: 0.4689
   F1-Score: 0.3838

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4808    0.8426    0.6122      5209
        REAL     0.3881    0.0989    0.1576      5260

    accuracy                         0.4689     10469
   macro avg     0.4344    0.4707    0.3849     10469
weighted avg     0.4342    0.4689    0.3838     10469


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      4389   820
   REAL      4740   520

❌ Errors: 5560 (53.11%)

🎯 Benchmarking on: SCRAPED.csv
==================================================
🔮 Predicting 36787 samples in batches of 16...
📊 Results for SCRAPED.csv:
   Total samples: 36787
   Accuracy: 0.5164
   Precision: 0.4783
   Recall: 0.5164
   F1-Score: 0.4903

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.5928    0.7123    0.6471     22898
        REAL     0.2896    0.1933    0.2318     13889

    accuracy                         0.5164     36787
   macro avg     0.4412    0.4528    0.4395     36787
weighted avg     0.4783    0.5164    0.4903     36787


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      16310  6588
   REAL      11204  2685

❌ Errors: 17792 (48.36%)

🎯 Benchmarking on: Trimmed-Kaggle1.csv
==================================================
🔮 Predicting 2901 samples in batches of 16...
📊 Results for Trimmed-Kaggle1.csv:
   Total samples: 2901
   Accuracy: 0.4619
   Precision: 0.4018
   Recall: 0.4619
   F1-Score: 0.3762

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4837    0.8249    0.6098      1479
        REAL     0.3166    0.0844    0.1333      1422

    accuracy                         0.4619      2901
   macro avg     0.4002    0.4546    0.3716      2901
weighted avg     0.4018    0.4619    0.3762      2901


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      1220   259
   REAL      1302   120

❌ Errors: 1561 (53.81%)

🎯 Benchmarking on: trimmed-scraped.csv
==================================================
🔮 Predicting 2879 samples in batches of 16...
📊 Results for trimmed-scraped.csv:
   Total samples: 2879
   Accuracy: 0.4463
   Precision: 0.4131
   Recall: 0.4463
   F1-Score: 0.4008

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4751    0.7148    0.5708      1483
        REAL     0.3472    0.1612    0.2202      1396

    accuracy                         0.4463      2879
   macro avg     0.4112    0.4380    0.3955      2879
weighted avg     0.4131    0.4463    0.4008      2879


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      1060   423
   REAL      1171   225

❌ Errors: 1594 (55.37%)

🎯 Benchmarking on: trimmed-WEL.csv
==================================================
🔮 Predicting 2963 samples in batches of 16...
📊 Results for trimmed-WEL.csv:
   Total samples: 2963
   Accuracy: 0.4641
   Precision: 0.4644
   Recall: 0.4641
   F1-Score: 0.3760

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.4640    0.8643    0.6038      1400
        REAL     0.4648    0.1056    0.1721      1563

    accuracy                         0.4641      2963
   macro avg     0.4644    0.4849    0.3879      2963
weighted avg     0.4644    0.4641    0.3760      2963


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      1210   190
   REAL      1398   165

❌ Errors: 1588 (53.59%)

 Benchmarking on: WELFake_Dataset.csv
==================================================
🔮 Predicting 62313 samples in batches of 16...
📊 Results for WELFake_Dataset.csv:
   Total samples: 62313
   Accuracy: 0.5078
   Precision: 0.4361
   Recall: 0.5078
   F1-Score: 0.4221

📈 Detailed Classification Report:
              precision    recall  f1-score   support

        FAKE     0.5340    0.8512    0.6563     34404
        REAL     0.3154    0.0845    0.1333     27909

    accuracy                         0.5078     62313
   macro avg     0.4247    0.4679    0.3948     62313
weighted avg     0.4361    0.5078    0.4221     62313


🎭 Confusion Matrix:
   True\Pred  FAKE  REAL
   FAKE      29284  5120
   REAL      25550  2359

❌ Errors: 30670 (49.22%)

"""