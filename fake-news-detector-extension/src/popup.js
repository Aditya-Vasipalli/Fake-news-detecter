document.addEventListener('DOMContentLoaded', function () {
    const analyzeButton = document.getElementById('analyze-button');
    const resultDiv = document.getElementById('result');

    analyzeButton.addEventListener('click', async () => {
        const textInput = document.getElementById('text-input').value;

        if (textInput) {
            resultDiv.textContent = 'Analyzing...';

            try {
                const response = await fetch('http://localhost:5000/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: textInput }),
                });

                const data = await response.json();
                resultDiv.textContent = `Prediction: ${data.prediction}, Confidence: ${data.confidence}`;
            } catch (error) {
                resultDiv.textContent = 'Error analyzing text. Please try again.';
            }
            
        } else {
            resultDiv.textContent = 'Please enter some text to analyze.';
        }
    });
});