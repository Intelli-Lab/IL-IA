import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Charger le modèle et les scalers
model = load_model("/var/www/html/plugins/script/data/modele_ia_chauffage.h5")
scaler_x = joblib.load("/var/www/html/plugins/script/data/scaler_x.gz")
scaler_y = joblib.load("/var/www/html/plugins/script/data/scaler_y.gz")

# Lire les entrées depuis input.json
with open("/var/www/html/plugins/script/data/input.json") as f:
    donnees = json.load(f)

features_order = [
    "presence", "heure", "jour_semaine", "temperature_exterieure", "temperature_interieure",
    "humidite_interieure", "ensoleillement", "nombre_personnes", "type_piece",
    "temperature_prevue_1h", "exposition", "huisserie", "isolation"
]

X_input = np.array([[donnees[f] for f in features_order]])
X_scaled = scaler_x.transform(X_input)

# Prédiction et retour à l'échelle réelle
y_scaled = model.predict(X_scaled)
consigne = scaler_y.inverse_transform(y_scaled)[0][0]
chauffage = "ON" if consigne > donnees["temperature_interieure"] else "OFF"

# Sauvegarder
with open("/var/www/html/plugins/script/data/output.json", "w") as f:
    json.dump({
        "consigne": round(float(consigne), 2),
        "chauffage": chauffage
    }, f, indent=2)

print(f"✅ Consigne : {consigne:.2f}°C — Chauffage : {chauffage}")
