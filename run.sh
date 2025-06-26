#!/usr/bin/with-contenv bashio

python3 -m venv .venv

# Activer l'environement
source ./.venv/bin/activate

pip install -r require.txt

python3 entrainement-IA.py
