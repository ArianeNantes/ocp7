#!/bin/bash

# Emplacement racine de l'app
APP_ROOT=/home/site/wwwroot

# Nom de l'environnement virtuel
VENV_NAME=antenv

# Chemin complet de l'environnement virtuel
VENV_PATH=$APP_ROOT/$VENV_NAME

echo "=== Startup script démarré ==="

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "$VENV_PATH" ]; then
    echo "Création de l'environnement virtuel $VENV_NAME..."
    python3 -m venv $VENV_PATH
fi

# Activer l'environnement virtuel
source $VENV_PATH/bin/activate

# Installer les dépendances
echo "Installation des dépendances depuis requirements.txt..."
pip install --upgrade pip
pip install -r $APP_ROOT/requirements.txt

# Lancer l'application avec gunicorn
echo "Démarrage de l'application..."
exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind=0.0.0.0:8000