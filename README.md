# Fake News Detector Chrome Extension

## Overview
This Chrome extension allows you to detect fake news on any news article page. It uses a machine learning model (Logistic Regression with TF-IDF) and SHAP explainability to highlight the most influential words in the article, showing which words push the prediction towards "Fake" or "Real" news.

- **Red highlights**: Words pushing the prediction toward "Fake" (intensity = SHAP value)
- **Green highlights**: Words pushing the prediction toward "Real" (intensity = SHAP value)

![image](https://github.com/user-attachments/assets/348d829b-1065-4a7f-914b-4ae689608567)
Demo image taken from <a href="https://indianexpress.com/article/business/aviation/india-cancels-security-clearance-turkish-celebi-aviation-security-airports-10008586/ ">The Indian Express</a>, used for academic demonstration purposes only.


## How it Works
1. **Backend**: A FastAPI server runs locally, serving a `/predict` endpoint that takes a news article URL, predicts if it's fake or real, and returns SHAP explanations. It also serves a `/plot` endpoint for the SHAP bar plot image.
2. **Extension**: The extension popup autofills the current tab's URL. When you click "Analyze", it sends the URL to the backend, displays the prediction, and highlights the most influential words on the page.

## Setup Instructions

### 1. Install Python Dependencies
In your project root, run:
```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend
In your project root, run:
```bash
python -m uvicorn shap_api:app --reload
```
- The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Test it at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Load the Chrome Extension
1. Go to `chrome://extensions/` in Chrome.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked** and select the `fake-news-detector-extension` folder.
4. Pin the extension for easy access.

### 4. Use the Extension
- Navigate to any news article.
- Click the extension icon.
- The popup will autofill the current URL. Click **Analyze**.
- The extension will display the prediction and highlight the most influential words on the page.

## File Structure
```
fake-news-detector-extension/
├── icon.jpg
├── manifest.json
├── popup.html
├── popup.js
├── LICENSE
├── README.md
└── src/
    ├── background.js
    ├── content.js  # Highlights words on the page
```
##DEPLOYMENT:
- Model available at: https://huggingface.co/aditya-vasipalli/fake-news-detector
    
## Notes
- The backend must be running for the extension to work.
- The extension only works on news article pages (not on all websites).

## License
See [LICENSE](LICENSE).
