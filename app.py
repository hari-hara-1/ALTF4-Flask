from flask import Flask, request, jsonify, render_template
import joblib
import os
from evaluatemodel import evaluate_credit_score

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "credit_score_model.pkl")
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json  # Expect JSON input
    score, suggestions = evaluate_credit_score(user_input, model, return_suggestions=True)
    return jsonify({
        'credit_score': score,
        'suggestions': suggestions
    })

if __name__ == '__main__':
    app.run(debug=True)
