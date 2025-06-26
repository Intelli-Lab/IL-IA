from flask import *
import prediction_chauffage as pc

app = Flask(__name__, template_folder='html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
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
    consigne = pc.compute(params) # Il va faloir encore traiter les params pour les adapter.
    return jsonify({"consigne": consigne})