import requests
import json

def test_api():
    """Test the updated SHAP API with sliding window functionality"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Fake News Detector API with Sliding Window")
    print("=" * 60)
    
    # Test 1: Short text (should use standard approach)
    print("\n1. Testing SHORT text:")
    short_text = "Breaking: Scientists discover new renewable energy breakthrough"
    
    try:
        response = requests.get(f"{base_url}/predict", params={"text": short_text})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence_real']:.3f}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 2: Long text (should use sliding window)
    print("\n2. Testing LONG text (should trigger sliding window):")
    long_text = """
    Breaking News: Revolutionary Solar Technology Breakthrough at MIT. 
    Scientists at the Massachusetts Institute of Technology have announced a groundbreaking discovery 
    that could revolutionize the renewable energy industry. The research team, led by Dr. Sarah Johnson, 
    has developed a new type of solar panel that achieves unprecedented efficiency rates of over 95%, 
    far exceeding current commercial solar panels which typically reach only 20-22% efficiency.
    
    The breakthrough comes from a novel approach using quantum dots embedded in perovskite materials. 
    These quantum dots can capture photons across a much broader spectrum of light, including infrared 
    radiation that traditional solar panels cannot utilize effectively. "This represents the most 
    significant advancement in photovoltaic technology since the invention of the silicon solar cell," 
    stated Dr. Johnson during a press conference held at MIT's campus yesterday.
    
    The new panels, which the team has dubbed "Quantum Enhanced Photovoltaic Cells" or QEPCs, have 
    undergone rigorous testing over the past two years. Initial laboratory results were promising, 
    but the researchers needed to verify that the technology could perform effectively in real-world 
    conditions across various climates and weather patterns.
    
    Field tests were conducted in multiple locations ranging from the sunny deserts of Arizona to 
    the frequently cloudy regions of Northern Europe. Remarkably, the QEPCs consistently showed 
    efficiency rates above 90% across all tested environments, demonstrating their versatility and 
    reliability under diverse conditions.
    
    One of the most impressive aspects of this new technology is its performance during low-light 
    conditions. Traditional solar panels experience significant decreases in efficiency during cloudy 
    weather or during early morning and late evening hours when sunlight is limited. However, the 
    QEPCs maintain over 70% of their peak efficiency even under these challenging conditions, thanks 
    to their ability to harness infrared radiation and effectively capture scattered light.
    """ * 3  # Make it even longer to definitely trigger sliding window
    
    try:
        response = requests.post(f"{base_url}/predict", json={"text": long_text})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence_real']:.3f}")
            print(f"Keywords: {len(result.get('top_keywords', []))} found")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")
    
    # Test 3: URL analysis (if endpoint exists)
    print("\n3. Testing URL analysis:")
    test_url = "https://www.thehindu.com/news/international/trump-ready-for-phase-two-of-russia-sanctions-over-ukraine-conflict/article70023637.ece"
    
    try:
        response = requests.get(f"{base_url}/analyze_url", params={"url": test_url})
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"Prediction: {result.get('prediction', 'N/A')}")
            print(f"Confidence: {result.get('confidence_real', 'N/A')}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
