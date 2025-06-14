# Utilise une image de base Python légère
FROM python:3.9-slim

# Crée un dossier de travail dans le conteneur
WORKDIR /app

# Copie les dépendances et les installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie uniquement les fichiers nécessaires à l'API
COPY main.py .
COPY models/best_model_lgbm.pkl models/best_model_lgbm.pkl
COPY data/cleaned/X_new_loans_lgbm.csv data/X_new_loans_lgbm.csv
COPY src/api/ src/api

# Expose le port utilisé par uvicorn
EXPOSE 8000

# Commande pour lancer l'application FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
