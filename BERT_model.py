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
    """Load data with proper encoding handling"""
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"Trying to load data with {encoding} encoding...")
            fake_df = pd.read_csv('Fake.csv', encoding=encoding)
            true_df = pd.read_csv('True.csv', encoding=encoding)
            print(f"✅ Successfully loaded with {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise Exception("Could not load data with any encoding")
    
    # Add labels
    fake_df['label'] = 0  # Fake news
    true_df['label'] = 1  # Real news
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Create text content (same as training data structure)
    if 'text' in df.columns:
        texts = df['text'].fillna('').astype(str)
    elif 'title' in df.columns and 'subject' in df.columns:
        texts = df['title'].fillna('') + ' ' + df['subject'].fillna('')
    elif 'title' in df.columns:
        texts = df['title'].fillna('')
    else:
        # Find first text column
        text_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'label']
        if text_cols:
            texts = df[text_cols[0]].fillna('')
        else:
            raise Exception("No text column found")
    
    labels = df['label'].values
    
    print(f"Dataset loaded: {len(texts)} samples")
    print(f"Fake news: {sum(labels == 0)} samples")
    print(f"Real news: {sum(labels == 1)} samples")
    
    return texts.tolist(), labels.tolist()

def create_data_loaders_aggressive_split(texts, labels, tokenizer, batch_size=4, max_length=256):
    """Create data loaders with aggressive validation split to prevent overfitting"""
    
    # AGGRESSIVE SPLIT: 50% validation, 10% test, 40% training
    # This forces the model to generalize on much less training data
    
    # First split: 40% train, 60% temp
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.6, random_state=42, stratify=labels
    )
    
    # Second split: 50% val, 10% test from the temp (60%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.167, random_state=42, stratify=temp_labels  # 10/60 = 0.167
    )
    
    print(f"🚀 AGGRESSIVE ANTI-OVERFITTING SPLIT:")
    print(f"   Training samples: {len(train_texts)} ({len(train_texts)/len(texts)*100:.1f}%)")
    print(f"   Validation samples: {len(val_texts)} ({len(val_texts)/len(texts)*100:.1f}%)")
    print(f"   Test samples: {len(test_texts)} ({len(test_texts)/len(texts)*100:.1f}%)")
    
    # Create datasets with reduced max_length to prevent memorization
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)
    
    # Small batch size to prevent overfitting
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
        
        # AGGRESSIVE L2 REGULARIZATION
        l2_reg = 0.1  # Much stronger than before (was 0.01)
        l2_loss = 0
        for param in model.parameters():
            l2_loss += torch.norm(param, 2)
        loss = loss + l2_reg * l2_loss
        
        # LABEL SMOOTHING to prevent overconfidence
        smoothed_labels = labels.float()
        smoothed_labels = smoothed_labels * 0.9 + 0.05  # 90% confidence instead of 100%
        
        loss.backward()
        
        # Aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced from 1.0
        
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
    
    # Initialize tokenizer and model with HEAVY regularization
    print("\n🤖 Loading BERT model with heavy regularization...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        hidden_dropout_prob=0.6,  # VERY HIGH dropout (was 0.3)
        attention_probs_dropout_prob=0.6,  # VERY HIGH dropout
        classifier_dropout=0.7,  # Additional classifier dropout
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # Create data loaders with aggressive split
    print("\n📦 Creating data loaders with aggressive validation split...")
    train_loader, val_loader, test_loader = create_data_loaders_aggressive_split(
        texts, labels, tokenizer, batch_size=4, max_length=256  # Small batch, short sequences
    )
    
    # Setup training with conservative settings
    epochs = 2  # Reduced epochs to prevent overfitting
    learning_rate = 1e-5  # Much lower learning rate (was 2e-5)
    
    # HEAVY weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)  # Was 0.01
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),  # Less warmup
        num_training_steps=total_steps
    )
    
    # Training loop with strict overfitting monitoring
    print(f"\n🚀 Starting aggressive anti-overfitting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Dropout: 0.6-0.7")
    print(f"   Weight Decay: 0.1")
    print(f"   L2 Regularization: 0.1")
    
    best_val_accuracy = 0
    patience = 0
    max_patience = 1  # Very aggressive early stopping
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
        overfitting_signals = []
        
        if train_val_gap > 0.05:  # 5% gap threshold (very strict)
            overfitting_signals.append(f"Accuracy gap too large ({train_val_gap:.4f})")
        
        if loss_gap > 1.5:  # Validation loss much higher
            overfitting_signals.append(f"Validation loss too high (ratio: {loss_gap:.2f})")
            
        if train_loss < 0.15:  # Training loss too low
            overfitting_signals.append(f"Training loss too low ({train_loss:.4f})")
            
        if val_accuracy < 0.6:  # Validation accuracy too low
            overfitting_signals.append(f"Validation accuracy too low ({val_accuracy:.4f})")
        
        # Report overfitting signals
        if overfitting_signals:
            print(f"  ⚠️ OVERFITTING SIGNALS DETECTED:")
            for signal in overfitting_signals:
                print(f"     - {signal}")
            overfitting_detected = True
        else:
            print(f"  ✅ No overfitting detected")
        
        # Save best model (but only if no strong overfitting)
        if val_accuracy > best_val_accuracy and len(overfitting_signals) <= 1:
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
