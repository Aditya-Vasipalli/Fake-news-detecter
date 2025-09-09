import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

def test_bert_model():
    """Test the trained BERT model with real examples"""
    
    # Check if model files exist
    try:
        # Load the saved model
        model = BertForSequenceClassification.from_pretrained('bert_model')
        tokenizer = BertTokenizer.from_pretrained('bert_model')
        
        # Load model state
        checkpoint = torch.load('fake_news_bert_model.pth', map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Load metadata if available
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
                print(f"📊 Best validation accuracy: {metadata.get('best_val_accuracy', 'Unknown')}")
        except:
            print("📊 No metadata file found")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with various examples
    test_examples = [
        # Likely real news
        "The Federal Reserve announced today that it will maintain interest rates at current levels through the end of 2024.",
        "Scientists at MIT published new research showing improvements in solar panel efficiency using perovskite materials.",
        "The unemployment rate decreased to 3.7% in the latest monthly jobs report from the Bureau of Labor Statistics.",
        
        # Likely fake news patterns
        "SHOCKING: Doctors hate this one simple trick that melts belly fat overnight!",
        "BREAKING: Aliens land in Area 51, government tries to cover it up with fake news!",
        "This mom discovered one weird trick that makes you rich in 30 days - banks hate her!",
        "UNBELIEVABLE: Celebrity dies and comes back to life, reveals shocking secrets!",
        
        # Ambiguous/tricky examples
        "Local man wins lottery, quits job to become professional gamer",
        "New study suggests coffee may help prevent certain types of cancer",
        "Celebrity couple announces divorce after 20 years of marriage"
    ]
    
    print("\n🧪 Testing model with various examples:")
    print("=" * 80)
    
    fake_count = 0
    real_count = 0
    
    for i, text in enumerate(test_examples, 1):
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Interpret results
        label = "Fake News" if predicted_class == 0 else "Real News"
        if predicted_class == 0:
            fake_count += 1
        else:
            real_count += 1
        
        print(f"{i:2d}. {label} (Confidence: {confidence:.3f})")
        print(f"    Text: {text[:70]}{'...' if len(text) > 70 else ''}")
        print()
    
    print("=" * 80)
    print(f"📈 Results Summary:")
    print(f"   Predicted as Fake News: {fake_count}/{len(test_examples)}")
    print(f"   Predicted as Real News: {real_count}/{len(test_examples)}")
    
    # Check if model is just predicting everything as one class
    if fake_count == 0:
        print("⚠️  WARNING: Model predicts EVERYTHING as Real News - likely overfitted!")
    elif real_count == 0:
        print("⚠️  WARNING: Model predicts EVERYTHING as Fake News - likely overfitted!")
    elif fake_count == len(test_examples) or real_count == len(test_examples):
        print("⚠️  WARNING: Model only predicts one class - definitely overfitted!")
    else:
        print("✅ Model shows balanced predictions - likely working correctly!")
    
    return fake_count, real_count

if __name__ == "__main__":
    test_bert_model()
