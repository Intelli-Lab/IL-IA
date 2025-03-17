from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import csv

app = Flask(__name__)

# Ici on stock les donnés et seuls les dernieres sont envoyées (celle de la journée passé)

# Exemple de données (à remplacer par des données réelles)
data = {
    'temperature': [15, 16, 17, 18, 19],
    'heating_usage': [2.5, 2.6, 2.7, 2.8, 2.9]
}
df = pd.DataFrame(data)

# Séparer les données en ensembles d'entraînement et de test
X = df[['temperature']]
y = df['heating_usage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['GET'])
def predict():
    temperature = request.args.get('temperature', type=float)
    if temperature is None:
        return jsonify({"error": "Missing temperature parameter"}), 400

    predicted_usage = model.predict([[temperature]])
    return jsonify({"predicted_usage": predicted_usage[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8124)
