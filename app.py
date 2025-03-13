from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import math


df = pd.read_csv("synthetic_transactions.csv")

features = df[['amount', 'gas_fee', 'transaction_count', 'wallet_age']]
labels = df['is_fraud']

clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(features)

preds = clf.predict(features)
preds_binary = np.where(preds == -1, 1, 0)

# Evaluate the model
precision = precision_score(labels, preds_binary)
recall = recall_score(labels, preds_binary)
f1 = f1_score(labels, preds_binary)

print("Model Evaluation:")
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))


joblib.dump(clf, 'isolation_forest_model.pkl')


app = Flask(__name__)

def decision_score_to_probability(score):
  
    prob = 1 / (1 + math.exp(score))
    return prob

@app.route('/detect_fraud', methods=['POST'])
def detect_fraud():
    """
    Expects JSON input with the following fields:
    {
        "amount": <float>,
        "gas_fee": <float>,
        "transaction_count": <int>,
        "wallet_age": <int>
    }
    """
    data = request.get_json(force=True)
    
    try:
        # Extract the required features from the request
        input_features = np.array([[data['amount'], data['gas_fee'], data['transaction_count'], data['wallet_age']]])
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    
    # Compute the decision function score
    score = clf.decision_function(input_features)[0]
    # Convert score to a fraud probability using a logistic transformation
    fraud_prob = decision_score_to_probability(score)
    
    return jsonify({"fraud_probability": fraud_prob})

@app.route('/status', methods=['GET'])
def status():
    # Return model training status and evaluation metrics
    return jsonify({
        "status": "Model trained successfully",
        "evaluation": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    })

if __name__ == '__main__':
    # Run the Flask API
    app.run(debug=True)
