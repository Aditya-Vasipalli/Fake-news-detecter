from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0].tolist()
    return jsonify({'prediction': int(pred), 'proba': proba})

if __name__ == '__main__':
    app.run(port=5000)