# src/services/predictor.py
import joblib
import os
import numpy as np
from pathlib import Path
import pandas as pd
from src.constantes import MODEL_DIR, DATA_CLEAN_DIR

# Chargement du modèle
"""# On remonte d'un cran, puis on redescend vers models/
# __file__ contient le chemin du fichier en cours d'exécution
# Donc chemin relatif garanti.
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_lgbm.pkl")
# Optionel de passer en chemin absolu, mais apparemment bonne pratique.
# Car plus robuste si on change de rép de travail de départ, quelque fois bugs subtiles...
# Mieux quand on n'est plus cantonné à jupyter etc. mais dans le cadre d'une api.
model_path = os.path.abspath(model_path)
model = joblib.load(model_path)
"""

model = None  # Défini plus tard

# On a tout mis dans test_api
# On remonte de deux crans et on descend dans models
model_dir = os.path.join(os.path.dirname(__file__), "..", "..", MODEL_DIR)
# On remonte de deux crans et on descend dans data/cleaned
data_dir = os.path.join(os.path.dirname(__file__), "..", "..", DATA_CLEAN_DIR)
# input_dir = os.path.join(os.path.dirname(__file__), "..", "..", "test_api")


# Service qui charge le modèle et le jdd de test
def load_model():
    global model
    model_path = os.path.join(model_dir, "best_model_lgbm.pkl")
    model_path = os.path.abspath(model_path)
    model = joblib.load(model_path)
    return model


def load_data_test():
    global df
    df_path = os.path.join(data_dir, "X_new_loans_lgbm.csv")
    df_path = os.path.abspath(df_path)
    dtypes_path = os.path.join(data_dir, "dtypes_lgbm.pkl")
    dtypes_path = os.path.abspath(dtypes_path)
    dic_dtypes = joblib.load(dtypes_path)
    df = pd.read_csv(df_path, dtype=dic_dtypes)
    if "SK_ID_CURR" in df.columns:
        df.set_index("SK_ID_CURR", inplace=True)  # pour accès rapide
    if "TARGET" in df.columns:
        df = df.drop(columns=["TARGET"])
    df.sort_index()
    return df


# Service qui retourne les données correspondant à l'ID_client (SK_ID_CURR) en
# Lisant dans le fichier test.csv
def get_features_from_id(client_id: int):
    if client_id not in df.index:
        return None
    features = df.loc[client_id].to_dict()

    # Comme on ne peut pas mettre des np.NaN dans un json,
    # On rempace ces valeurs par des None (qui feront des null en JSON)
    cleaned = {}
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            cleaned[k] = (
                None  # None donnera null en JSON et pourra être traité comme un NaN dans lightgbm
            )
        else:
            cleaned[k] = v
    return cleaned


def load_threshold():
    global threshold
    path_threshold = os.path.join(model_dir, "best_threshold_lgbm.pkl")
    threshold = joblib.load(path_threshold)
    return threshold


# service qui retourne la probabilité et la prédiction en fonction des données client et du modèle
def predict_from_features(features_dict):
    # Les null du JSON vont créer des types Object dans le DataFrame,
    # On retransforme de null vers np.nan
    cleaned = {}
    for k, v in features_dict.items():
        if v is None or v == "null":
            cleaned[k] = np.nan
        else:
            cleaned[k] = v
    # On fait un dataframe et non juste un array, car le modèle a été entrainé sur un dataframe avec des noms de colonnes,
    # sinon on a un warning à cause des noms de colonnes manuqnats
    X = pd.DataFrame([cleaned])
    proba = model.predict_proba(X)[0][1]
    if proba > threshold:
        prediction = 1
    else:
        prediction = 0
    return proba, prediction, threshold


# service qui retourne la prédiction en fonction des données client et du modèle
def predict_proba_from_features(features_dict):
    # Les null du JSON vont créer des types Object dans le DataFrame,
    # On retransforme de null vers np.nan
    cleaned = {}
    for k, v in features_dict.items():
        if v is None or v == "null":
            cleaned[k] = np.nan
        else:
            cleaned[k] = v
    # On fait un dataframe et non juste un array, car le modèle a été entrainé sur un dataframe avec des noms de colonnes,
    # sinon on a un warning à cause des noms de colonnes manuqnats
    X = pd.DataFrame([cleaned])
    # Prédiction de la probabilité de la classe "default" (classe 1)
    # Attention : première ligne (=0), deuxième colonne (=1) (pas comme en np ni cudf habituels)
    proba_default = model.predict_proba(X)[0][1]
    return proba_default
