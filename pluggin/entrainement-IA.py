import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Charger les données
with open("/var/www/html/plugins/script/data/donnees_ia_chauffage_ameliorees.json") as f:
    data = pd.DataFrame(json.load(f))

X = data.drop(columns=["temperature_consigne"])
y = data["temperature_consigne"].values.reshape(-1, 1)

# Normalisation
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Sauvegarder les scalers
joblib.dump(scaler_x, "/var/www/html/plugins/script/data/scaler_x.gz")
joblib.dump(scaler_y, "/var/www/html/plugins/script/data/scaler_y.gz")

# Séparer les données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Modèle
model = Sequential([
    Dense(64, activation="relu", input_shape=(X.shape[1],)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=120, verbose=1)

# Sauvegarde
model.save("/var/www/html/plugins/script/data/modele_ia_chauffage.h5")

# Évaluation (en degrés réels)
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mse = np.mean((y_pred - y_true) ** 2)
print(f"✅ Modèle corrigé — MSE réel : {mse:.2f}")
