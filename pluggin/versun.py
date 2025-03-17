from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# # Chargement du modèle TensorFlow (à entraîner au préalable)
# model = tf.keras.models.load_model('path_to_your_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données JSON de la requête
    data = request.get_json(force=True)

    # Prétraitement des données 
    weather_forecast = np.array(data['weather_forecast'])
    historical_data = np.array(data['historical_data'])

    # Prédiction avec le modèle
    prediction = model.predict([weather_forecast, historical_data])

    # Retourne la prédiction sous forme de JSON
    return jsonify({'heating_schedule': prediction.tolist()})

@app.route('/learn', methods=['POST'])
def learn():
    # Récupération des données JSON de la requête
    data = request.get_json(force=True)

    # Extraction des nouvelles données d'entraînement
    new_weather_data = np.array(data['new_weather_data'])
    new_heating_data = np.array(data['new_heating_data'])

    # Prétraitement des données (exemple simplifié)
    # Vous devrez adapter cela en fonction de vos besoins spécifiques
    X_train = new_weather_data
    y_train = new_heating_data

    # Entraînement du modèle avec les nouvelles données
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Sauvegarde du modèle mis à jour
    model.save('path_to_your_model.h5')

    # Retourne un message de succès
    return jsonify({'message': 'Model successfully updated with new data.'})

if __name__ == '__main__':
    app.run(debug=True)
