#!/bin/bash

python3 -m venv .venv

# Activer l'environement
source ./.venv/bin/activate

pip install -r require.txt

python3 prediction_ia_chauffage.py