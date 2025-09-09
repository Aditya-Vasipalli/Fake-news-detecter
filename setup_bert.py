#!/usr/bin/env python3
"""
Setup script to train BERT model and prepare for API usage
"""

import os
import sys

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        print("✅ PyTorch and Transformers are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = ["True.csv", "Fake.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        return False
    else:
        print("✅ Data files found")
        return True

def train_bert_model():
    """Train the BERT model"""
    print("🚀 Starting BERT model training...")
    print("This may take several minutes depending on your hardware...")
    
    try:
        exec(open('bert.py').read())
        print("✅ BERT model training completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def test_api():
    """Test if the API can be imported and initialized"""
    try:
        print("🧪 Testing API import...")
        # Try importing the API module
        import shap_api
        print("✅ API can be imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def main():
    print("🔧 Setting up BERT-based Fake News Detector")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return False
    
    # Check data files
    if not check_data_files():
        return False
    
    # Train BERT model
    if not train_bert_model():
        return False
    
    # Test API
    if not test_api():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the API: python shap_api.py")
    print("2. Test your Chrome extension")
    print("\nThe API will be available at: http://127.0.0.1:8000")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
