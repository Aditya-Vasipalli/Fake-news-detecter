import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm
import json
import os
import warnings
warnings.filterwarnings('ignore')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduced max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data_with_encoding():
    """Load data from multiple sources with proper encoding handling"""
    all_texts = []
    all_labels = []
    
    print("📚 Loading data from multiple sources...")
    
    # ============================================================================
    # 1. LOAD ORIGINAL FAKE/TRUE DATASET
    # ============================================================================
    print("\n1️⃣ Loading original Fake.csv and True.csv...")
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    fake_df = None
    true_df = None
    
    for encoding in encodings:
        try:
            print(f"   Trying {encoding} encoding...")
            fake_df = pd.read_csv('Fake.csv', encoding=encoding)
            true_df = pd.read_csv('True.csv', encoding=encoding)
            print(f"   ✅ Original dataset loaded with {encoding}")
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    if fake_df is not None and true_df is not None:
        # Add labels
        fake_df['label'] = 0  # Fake news
        true_df['label'] = 1  # Real news
        
        # Combine datasets
        original_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Extract text
        if 'text' in original_df.columns:
            orig_texts = original_df['text'].fillna('').astype(str).tolist()
        elif 'title' in original_df.columns:
            orig_texts = original_df['title'].fillna('').astype(str).tolist()
        else:
            text_cols = [col for col in original_df.columns if original_df[col].dtype == 'object' and col != 'label']
            if text_cols:
                orig_texts = original_df[text_cols[0]].fillna('').astype(str).tolist()
            else:
                orig_texts = []
        
        orig_labels = original_df['label'].tolist()
        
        all_texts.extend(orig_texts)
        all_labels.extend(orig_labels)
        print(f"   📊 Original dataset: {len(orig_texts)} samples (Fake: {sum(original_df['label']==0)}, Real: {sum(original_df['label']==1)})")
    
    # ============================================================================
    # 2. LOAD LIAR DATASET
    # ============================================================================
    print("\n2️⃣ Loading LIAR dataset...")
    
    try:
        # Load LIAR train set
        liar_path = "./liar_dataset-master/train.tsv"
        if os.path.exists(liar_path):
            liar_df = pd.read_csv(liar_path, sep='\t', header=None, encoding='utf-8')
            
            # LIAR format: ID, label, statement, subject, speaker, job, state, party, counts..., context
            liar_df.columns = ['id', 'label', 'statement', 'subject', 'speaker', 'job', 
                             'state', 'party', 'barely_true_count', 'false_count', 
                             'half_true_count', 'mostly_true_count', 'pants_fire_count', 'context']
            
            # Label mapping for LIAR dataset
            liar_label_mapping = {
                'true': 1,           # REAL
                'mostly-true': 1,    # REAL  
                'half-true': 0,      # FAKE (leaning false)
                'barely-true': 0,    # FAKE
                'false': 0,          # FAKE
                'pants-fire': 0      # FAKE
            }
            
            # Convert labels to binary
            liar_df['binary_label'] = liar_df['label'].map(liar_label_mapping)
            liar_df = liar_df.dropna(subset=['binary_label'])
            
            # Clean statements
            liar_df['statement'] = liar_df['statement'].astype(str).str.strip()
            liar_df = liar_df[liar_df['statement'].str.len() > 10]
            
            liar_texts = liar_df['statement'].tolist()
            liar_labels = liar_df['binary_label'].astype(int).tolist()
            
            all_texts.extend(liar_texts)
            all_labels.extend(liar_labels)
            print(f"   ✅ LIAR dataset loaded: {len(liar_texts)} samples (Fake: {sum(liar_df['binary_label']==0)}, Real: {sum(liar_df['binary_label']==1)})")
        else:
            print(f"   ⚠️ LIAR dataset not found at {liar_path}")
            
    except Exception as e:
        print(f"   ❌ Error loading LIAR dataset: {e}")
    
    # ============================================================================
    # 3. LOAD DATAVERSELL KEVIN WANG DATASETS
    # ============================================================================
    print("\n3️⃣ Loading DataverseNL Kevin Wang datasets...")
    
    dataset_folder = "./Kevin Wang/WEBAPP/datasets"
    if not os.path.exists(dataset_folder):
        dataset_folder = "./dataverseNL/Kevin Wang/WEBAPP/datasets"
    
    if os.path.exists(dataset_folder):
        dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]
        print(f"   Found {len(dataset_files)} CSV files")
        
        for filename in dataset_files:
            try:
                filepath = os.path.join(dataset_folder, filename)
                print(f"   📁 Processing {filename}...")
                
                # Try different encodings
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding)
                        break
                    except:
                        continue
                
                if df is None:
                    print(f"      ❌ Could not load {filename}")
                    continue
                
                # Auto-detect text and label columns
                text_column = None
                label_column = None
                
                # Find text column
                text_candidates = ['text', 'article_content', 'content', 'news', 'statement', 'title']
                for col in df.columns:
                    if any(candidate in col.lower() for candidate in text_candidates):
                        text_column = col
                        break
                
                if text_column is None:
                    # Use longest text column
                    text_lengths = {}
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            try:
                                avg_length = df[col].astype(str).str.len().mean()
                                text_lengths[col] = avg_length
                            except:
                                continue
                    if text_lengths:
                        text_column = max(text_lengths, key=text_lengths.get)
                
                # Find label column
                label_candidates = ['label', 'labels', 'fake', 'class', 'target', 'y']
                for col in df.columns:
                    if col.lower() in label_candidates:
                        label_column = col
                        break
                
                if not text_column or not label_column:
                    print(f"      ⚠️ Could not identify columns for {filename}")
                    continue
                
                # Standardize labels to binary (0=FAKE, 1=REAL)
                unique_labels = df[label_column].unique()
                label_mapping = {}
                
                for label in unique_labels:
                    label_str = str(label).lower().strip()
                    if label in [0, '0', 0.0] or 'fake' in label_str or 'false' in label_str:
                        label_mapping[label] = 0  # FAKE
                    elif label in [1, '1', 1.0] or 'real' in label_str or 'true' in label_str or 'reliable' in label_str:
                        label_mapping[label] = 1  # REAL
                    else:
                        # Default: first unique value is 0, second is 1
                        sorted_labels = sorted(unique_labels)
                        if label == sorted_labels[0]:
                            label_mapping[label] = 0
                        else:
                            label_mapping[label] = 1
                
                # Apply mapping
                df['binary_label'] = df[label_column].map(label_mapping)
                df = df.dropna(subset=['binary_label'])
                
                # Clean text
                df[text_column] = df[text_column].astype(str).str.strip()
                df = df[df[text_column].str.len() > 10]
                
                # Remove duplicates
                df = df.drop_duplicates(subset=[text_column])
                
                if len(df) > 0:
                    dataset_texts = df[text_column].tolist()
                    dataset_labels = df['binary_label'].astype(int).tolist()
                    
                    all_texts.extend(dataset_texts)
                    all_labels.extend(dataset_labels)
                    print(f"      ✅ Added {len(dataset_texts)} samples (Fake: {sum(df['binary_label']==0)}, Real: {sum(df['binary_label']==1)})")
                else:
                    print(f"      ⚠️ No valid samples in {filename}")
                    
            except Exception as e:
                print(f"      ❌ Error processing {filename}: {e}")
    else:
        print(f"   ⚠️ DataverseNL folder not found at {dataset_folder}")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n📊 TOTAL COMBINED DATASET:")
    print(f"   Total samples: {len(all_texts)}")
    print(f"   Fake news: {sum(1 for label in all_labels if label == 0)} samples")
    print(f"   Real news: {sum(1 for label in all_labels if label == 1)} samples")
    
    if len(all_texts) == 0:
        raise Exception("No data could be loaded from any source!")
    
    return all_texts, all_labels

def create_data_loaders_standard_split(texts, labels, tokenizer, batch_size=16, max_length=512):
    """Create data loaders with standard 80/10/10 split for proper learning"""
    
    # STANDARD SPLIT: 80% training, 10% validation, 10% test
    # This gives the model enough data to learn patterns properly
    
    # First split: 80% train, 20% temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Second split: 10% val, 10% test from the temp (20%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels  # 10/20 = 0.5
    )
    
    print(f"🚀 STANDARD DATA SPLIT FOR PROPER LEARNING:")
    print(f"   Training samples: {len(train_texts)} ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"   Validation samples: {len(val_texts)} ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"   Test samples: {len(test_texts)} ({len(test_texts)/len(texts)*100:.1f}%)")
    
    # Create datasets with full max_length for proper context
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Standard batch size for efficient training
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, device, desc="Evaluation"):
    """Evaluate model and return detailed metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=desc)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, accuracy, all_predictions, all_labels

def train_epoch_with_aggressive_regularization(model, train_loader, optimizer, scheduler, device, epoch):
    """Train one epoch with aggressive overfitting prevention"""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Standard training - no additional regularization
        # Let the model learn properly
        
        loss.backward()
        
        # Standard gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Standard value
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        accuracy = correct_predictions / total_predictions
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
        # STOP EARLY if loss gets too low (prevents memorization)
        if avg_loss < 0.2 and batch_idx > 50:  # Higher threshold than before
            print(f"\n🛑 STOPPING: Loss too low ({avg_loss:.4f}) - preventing overfitting!")
            break
    
    return total_loss / len(train_loader), correct_predictions / total_predictions

def main():
    print("🛡️ AGGRESSIVE ANTI-OVERFITTING BERT TRAINING")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Load data
    print("\n📊 Loading data...")
    texts, labels = load_data_with_encoding()
    
    # Initialize tokenizer and model with standard settings
    print("\n🤖 Loading BERT model with standard regularization...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        hidden_dropout_prob=0.1,  # Standard dropout for learning
        attention_probs_dropout_prob=0.1,  # Standard dropout for learning
        classifier_dropout=0.1,  # Gentle regularization
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Create data loaders with standard split
    print("\n📦 Creating data loaders with standard 80/10/10 split...")
    train_loader, val_loader, test_loader = create_data_loaders_standard_split(
        texts, labels, tokenizer, batch_size=16, max_length=512  # Standard batch size and max length
    )
    
    # Setup training with proper learning settings
    epochs = 4  # Proper number of epochs for learning
    learning_rate = 3e-5  # Standard BERT fine-tuning learning rate
    
    # Standard weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Standard value
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # Standard warmup
        num_training_steps=total_steps
    )
    
    # Training loop with proper learning settings
    print(f"\n🚀 Starting standard BERT fine-tuning...")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Dropout: 0.1 (standard)")
    print(f"   Weight Decay: 0.01 (standard)")
    print(f"   Batch Size: 16")
    print(f"   Max Length: 512")
    
    best_val_accuracy = 0
    patience = 0
    max_patience = 2  # Allow model to learn before stopping
    overfitting_detected = False
    
    training_stats = []
    
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        
        # Train
        train_loss, train_accuracy = train_epoch_with_aggressive_regularization(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # Validate
        val_loss, val_accuracy, val_predictions, val_labels = evaluate_model(
            model, val_loader, device, "Validation"
        )
        
        # Store stats
        epoch_stats = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }
        training_stats.append(epoch_stats)
        
        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Training Loss: {train_loss:.4f}, Training Acc: {train_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.4f}")
        
        # AGGRESSIVE overfitting detection
        train_val_gap = abs(train_accuracy - val_accuracy)
        loss_gap = val_loss / train_loss if train_loss > 0 else 1
        
        print(f"  📊 Train-Val Accuracy Gap: {train_val_gap:.4f}")
        print(f"  📊 Val/Train Loss Ratio: {loss_gap:.2f}")
        
        # Multiple overfitting checks
        # Standard overfitting check - only look for severe gaps
        overfitting_signals = []
        
        if train_val_gap > 0.15:  # 15% gap threshold - reasonable overfitting indicator
            overfitting_signals.append(f"Significant accuracy gap ({train_val_gap:.4f})")
        
        if loss_gap > 2.5:  # Only if validation loss is much higher than training
            overfitting_signals.append(f"Validation loss very high (ratio: {loss_gap:.2f})")
        
        # Report overfitting signals (but don't be overly restrictive)
        if overfitting_signals:
            print(f"  ⚠️ OVERFITTING SIGNALS DETECTED:")
            for signal in overfitting_signals:
                print(f"     - {signal}")
            overfitting_detected = True
        else:
            print(f"  ✅ No serious overfitting detected")
        
        # Save best model - allow learning even with minor overfitting signals
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience = 0
            
            print(f"  🎉 New best validation accuracy: {val_accuracy:.4f}")
            print(f"  💾 Saving model...")
            
            # Save model and tokenizer
            os.makedirs('bert_model', exist_ok=True)
            model.save_pretrained('bert_model')
            tokenizer.save_pretrained('bert_model')
            
            # Save model state
            torch.save(model.state_dict(), 'fake_news_bert_model.pth')
            
            # Save metadata
            metadata = {
                'model_name': 'bert-base-uncased',
                'num_labels': 2,
                'max_length': 256,
                'best_val_accuracy': float(val_accuracy),
                'epoch': epoch,
                'training_samples': len(train_loader.dataset),
                'validation_samples': len(val_loader.dataset),
                'test_samples': len(test_loader.dataset),
                'overfitting_detected': overfitting_detected,
                'train_val_gap': float(train_val_gap),
                'regularization': 'aggressive'
            }
            
            with open('model_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        else:
            patience += 1
            print(f"  📉 No improvement or overfitting detected. Patience: {patience}/{max_patience}")
            
            if patience >= max_patience or len(overfitting_signals) >= 2:
                print(f"  🛑 Early stopping: {'Multiple overfitting signals' if len(overfitting_signals) >= 2 else 'No improvement'}")
                break
    
    # Final test evaluation
    if os.path.exists('fake_news_bert_model.pth'):
        print(f"\n🧪 Final Test Evaluation...")
        test_loss, test_accuracy, test_predictions, test_labels = evaluate_model(
            model, test_loader, device, "Final Test"
        )
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  Best Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"  Final Test Accuracy: {test_accuracy:.4f}")
        print(f"  Final Test Loss: {test_loss:.4f}")
        
        print(f"\n📊 Detailed Test Set Results:")
        print(classification_report(test_labels, test_predictions, 
                                  target_names=['Fake News', 'Real News'], 
                                  digits=4))
        
        # Final overfitting check
        final_gap = abs(best_val_accuracy - test_accuracy)
        print(f"\n📈 Generalization Check:")
        print(f"  Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Gap: {final_gap:.4f}")
        
        if final_gap < 0.05:
            print(f"  ✅ EXCELLENT: Model generalizes well (gap < 5%)")
        elif final_gap < 0.1:
            print(f"  ⚠️ ACCEPTABLE: Some overfitting but manageable (gap < 10%)")
        else:
            print(f"  ❌ OVERFITTED: Significant overfitting detected (gap > 10%)")
    
    else:
        print(f"\n❌ No model was saved due to overfitting prevention")
    
    print(f"\n✅ Aggressive anti-overfitting training completed!")

if __name__ == "__main__":
    main()
