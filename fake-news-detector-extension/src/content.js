// This file contains the content script for the Chrome extension.
// It interacts with the web page to detect fake news articles.

const analyzeArticle = async (articleText) => {
    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: articleText }),
        });

        if (response.ok) {
            const result = await response.json();
            displayResult(result);
        } else {
            console.error('Error analyzing article:', response.statusText);
        }
    } catch (error) {
        console.error('Fetch error:', error);
    }
};

const displayResult = (result) => {
    const { prediction, confidence } = result;
    const message = `Prediction: ${prediction === 1 ? 'Fake' : 'Real'} (Confidence: ${confidence})`;
    
    const resultDiv = document.createElement('div');
    resultDiv.style.position = 'fixed';
    resultDiv.style.bottom = '10px';
    resultDiv.style.right = '10px';
    resultDiv.style.backgroundColor = 'white';
    resultDiv.style.border = '1px solid black';
    resultDiv.style.padding = '10px';
    resultDiv.style.zIndex = '1000';
    resultDiv.innerText = message;

    document.body.appendChild(resultDiv);
};

const articleText = document.body.innerText; // Get the text of the article
analyzeArticle(articleText); // Analyze the article text for fake news detection