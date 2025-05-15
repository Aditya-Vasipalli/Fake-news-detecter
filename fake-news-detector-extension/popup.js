document.addEventListener('DOMContentLoaded', function () {
  console.log('Popup loaded');
  const analyzeButton = document.getElementById('analyze-button');
  const resultDiv = document.getElementById('result');
  const urlInput = document.getElementById('url-input');

  // Autofill the URL input with the current tab's URL
  if (chrome && chrome.tabs) {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      if (tabs && tabs.length > 0) {
        urlInput.value = tabs[0].url;
        console.log('Autofilled URL:', tabs[0].url);
      }
    });
  }

  analyzeButton.addEventListener('click', async () => {
    const urlInputValue = urlInput.value.trim();
    const textInput = document.getElementById('text-input').value.trim();

    console.log('Analyze button clicked');
    console.log('URL input:', urlInputValue);
    console.log('Text input:', textInput);

    resultDiv.textContent = 'Analyzing...';

    try {
      let response, data;
      if (urlInputValue) {
        console.log('Sending request to /predict-url');
        response = await fetch('http://127.0.0.1:8000/predict-url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: urlInputValue })
        });
      } else if (textInput) {
        console.log('Sending request to /predict');
        response = await fetch('http://127.0.0.1:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: textInput })
        });
      } else {
        console.log('No input provided');
        resultDiv.textContent = 'Please enter a URL or some text.';
        return;
      }

      console.log('Response status:', response.status);
      if (response.ok) {
        data = await response.json();
        console.log('Response data:', data);
        if (data.title) {
          resultDiv.innerHTML = `<b>${data.title}</b><br>Prediction: <b>${data.prediction}</b><br>Confidence: Real ${data.confidence.real}, Fake ${data.confidence.fake}`;
        } else {
          resultDiv.innerHTML = `Prediction: <b>${data.prediction}</b><br>Confidence: Real ${data.confidence.real}, Fake ${data.confidence.fake}`;
        }
      } else {
        const errorText = await response.text();
        console.error('Error response:', errorText);
        resultDiv.textContent = 'Error: Could not analyze the article.';
      }
    } catch (error) {
      console.error('Fetch error:', error);
      resultDiv.textContent = 'Error: Could not connect to backend.';
    }
  });
});