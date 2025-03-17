import torch
import pandas as pd
from flask import Flask, request, jsonify, render_template
from torch import nn, optim
import os
import sys

# Définition des paramètres
TOLERANCE = 0.5
FIELDS_TEMP = ['temp_ext', 'temp_int', 'temp_consigne']
FIELDS_BOOL = ['mov']
FIELDS_OUTPUT = ['consigne']
FIELDS_LIST = FIELDS_TEMP + FIELDS_BOOL + FIELDS_OUTPUT
print("List of available fields :", FIELDS_LIST)

Model = None
Model_directory = 'models'
Default_model = 'V1.pt'
input_size = 5
# Définition du modèle d'IA

class ChauffageModel(nn.Module):
    def __init__(self, input_size):
        super(ChauffageModel, self).__init__()
        self.layer1 = nn.Linear(5, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x): # fonction forward qui prend en entrée un tenseur x et renvoie un tenseur x
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# Fonction d'entraînement
def entrainementLocal(data_path, model_path): # entraine avec un fichier préenregistre de données
    # Charger et prétraiter les données
    data = pd.read_csv(data_path)
    data = data[FIELDS_LIST]
    data[FIELDS_BOOL] = data[FIELDS_BOOL].astype(int)
    data = data.dropna()
    x = data[FIELDS_TEMP].values + data[FIELDS_BOOL].values
    y = data['consigne'].values
    # Entraîner le modèle
    model = ChauffageModel(5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for i in range(1000):
        optimizer.zero_grad()
        y_pred = model(torch.tensor(x, dtype=torch.float32))
        loss = criterion(y_pred, torch.tensor(y, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    # Sauvegarder le modèle
    torch.save(model, model_path)
    return model

# Fonction de décision
def decision(params): # prend en entrée les paramètres et le chemin du modèle
    # Charger le modèle si il y en a un
    if Model == None:
        model = torch.load(os.path.join(Model_directory, Default_model))
    else:
        model = Model
    # Prétraiter les données
    for key in FIELDS_TEMP:
        params[key] = [params[key]]
    for key in FIELDS_BOOL:
        params[key] = [int(params[key])]
    x = [params[key] for key in FIELDS_TEMP] + [params[key] for key in FIELDS_BOOL]
    # "Prédire" la consigne
    y_pred = model(torch.tensor(x, dtype=torch.float32))
    # Renvoyer la consigne
    return y_pred.item()

# API Flask
app = Flask(__name__, template_folder='html')

@app.route('/')
def api_root():
    return render_template('index.html')

@app.route('/interface', methods=['GET'])
def api_interface():
    return render_template('interface.html')

@app.route('/send_train_data_page', methods=['GET'])
def api_send_train_data_page():
    return render_template('train_data.html')

@app.route('/entrainement', methods=['POST'])
def api_entrainement():
    # Récupérer les données
    data_path = request.json['data_path']
    model_path = request.json['model_path']
    # Appeler la fonction d'entraînement
    entrainementLocal(data_path, model_path)
    return jsonify({"message": "Modèle entraîné."})

# /decision methodes POST & GET
@app.route('/decisionold', methods=['POST', 'GET'])
def api_decisionold():
    # Récupérer les paramètres
    params = {}
    if request.method == 'POST':
        for field in FIELDS_LIST:
            params[field] = request.json[field]
    if request.method == 'GET':
        for field in FIELDS_LIST:
            params[field] = request.args.get(field)
    for field in FIELDS_TEMP:
        if field not in params:
            return jsonify({"error": "Missing field: " + field})
        if not params[field].replace('.', '', 1).isdigit():
            return jsonify({"error": "Invalid value for field " + field})
        params[field] = float(params[field])
    for field in FIELDS_BOOL:
        if field not in params:
            return jsonify({"error": "Missing field: " + field})
        if params[field] == 'true':
            params[field] = 1
        elif params[field] == 'false':
            params[field] = 0 
        else:
            return jsonify({"error": "Invalid value for field " + field})
        params[field] = int(params[field])
    # Appeler la fonction de décision
    consigne = decision(params, 'model.pth')
    return jsonify({"consigne": consigne})

# decision pour json et web :
# on enregistre les champs dans un array en extrayant les valeurs des champs json ou web
@app.route('/decision', methods=['POST', 'GET'])
def api_decision():
    params = request.json if request.method == 'POST' else request.args.to_dict()
    for field in FIELDS_TEMP:
        if field not in params:
            return jsonify({"error": "Missing field: " + field})
        try:
            params[field] = float(params[field])
        except ValueError:
            return jsonify({"error": "Invalid value for field " + field})
    
    for field in FIELDS_BOOL:
        if field not in params:
            return jsonify({"error": "Missing field: " + field})
        if params[field].lower() == 'true':
            params[field] = 1
        elif params[field].lower() == 'false':
            params[field] = 0 
        else:
            return jsonify({"error": "Invalid value for field " + field})
        params[field] = int(params[field])
    
    consigne = decision(params)
    return jsonify({"consigne": consigne})

# model manager
@app.route('/model', methods=['POST', 'GET'])
def api_model():
    if request.method == 'GET':
        action = request.args.get('action')
        name = request.args.get('name')
    elif request.method == 'POST':
        action = request.json['action']
        name = request.json['name']
    if action == 'load':
        Model = torch.load(os.path.join(Model_directory, name))
        return jsonify({"message": "Model loaded."})
    elif action == 'save':
        torch.save(Model, os.path.join(Model_directory, name))
        return jsonify({"message": "Model saved."})
    elif action == 'list':
        models = os.listdir(Model_directory)
        return jsonify({"models": models})
    elif action == 'delete':
        os.remove(os.path.join(Model_directory, name))
        return jsonify({"message": "Model deleted."})
    elif action == 'reset':
        Model = ChauffageModel(input_size)
        # Convertir les paramètres du modèle en une liste de listes pour la sérialisation JSON
        parameters_serializable = [param.data.numpy().tolist() for param in Model.parameters()]
        return jsonify({
            "message": "Model reseted.",
            "parameters": parameters_serializable,
            "input_size": input_size, "output_size": 1,
            "layer detail :" : {
                "layer1": {
                    "input_size": 5,
                    "output_size": 128
                },
                "layer2": {
                    "input_size": 128,
                    "output_size": 64
                },
                "output_layer": {
                    "input_size": 64,
                    "output_size": 1
                }
            }
        })
    else:
        return jsonify({"error": "Invalid action."})

# Exécuter l'API
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)