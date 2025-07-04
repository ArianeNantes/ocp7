import numpy as np
import pandas as pd
import os

# import gc
import time

# import joblib
import subprocess
import io
from PIL import Image
import json
import requests
import shutil

# import optuna
# from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage
import mlflow
from mlflow.tracking import MlflowClient

from src.p7_constantes import HOST_MLFLOW, PORT_MLFLOW


# Renvoie True si la connexion à l'interface web est ok, False sinon.
# Vérifier que l'interface web est déjà démarrée avant de lancer un sous-processus
# nous évite de lancer plein de sous-processus pour rien.
# ui = User Interface
def is_mlflow_ui_started(host=HOST_MLFLOW, port=PORT_MLFLOW, verbose=True):
    mlflow_ui = f"http://{HOST_MLFLOW}:{PORT_MLFLOW}"
    mlflow.set_tracking_uri(mlflow_ui)
    try:
        response = requests.get(mlflow_ui)
        if response.status_code == 200:
            if verbose:
                print(f"L'interface web mlflow est disponible à l'adresse {mlflow_ui}")
            return True
    except requests.exceptions.RequestException as e:
        if verbose:
            print("La connexion MLFlow a échoué.")
            print("Error:", e)
        return False


# Démarre l'interface Web mlflow en ouvrant un sous-processus depuis un Notebook
# mlflow_tracking_uri="./mlruns" Les logs seront stockés dans le dossier courant, sous-dossier "mlruns"
def start_mlflow_ui(host=HOST_MLFLOW, port=PORT_MLFLOW, mlflow_tracking_uri="./mlruns"):
    # ui = User Interface (interface web de mlflow)
    mlflow_ui = f"http://{HOST_MLFLOW}:{PORT_MLFLOW}"
    if is_mlflow_ui_started(host=HOST_MLFLOW, port=PORT_MLFLOW, verbose=False):
        print(f"L'interface web mlflow est disponible à l'adresse {mlflow_ui}")
        return
    else:
        # uri = Uniform Resource Identifier, chemin vers l'endroit où mlflow stocke les donnée de suivi (runs, artifacts...) sur le disque
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        # Démarre MLflow UI via subprocess
        mlflow_ui_process = subprocess.Popen(
            [
                "mlflow",
                "ui",
                "--port",
                str(PORT_MLFLOW),
                "--backend-store-uri",
                mlflow_tracking_uri,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Donne un peu de temps au serveur pour démarrer
        time.sleep(5)
        is_mlflow_ui_started(host=HOST_MLFLOW, port=PORT_MLFLOW, verbose=True)
        return mlflow_ui_process


def create_or_load_experiment(
    name, description="", tags={}, artifact_location="mlflow_artifacts", verbose=True
):
    tags["mlflow.note.content"] = description

    # On vérifie si l'expérience existe déjà
    existing_experiment = mlflow.get_experiment_by_name(name)

    load_or_create = "Création de "
    if existing_experiment:
        experiment_id = existing_experiment.experiment_id
        load_or_create = "Chargement de "
    else:
        artifact_location = os.path.join(artifact_location, name)
        os.makedirs(artifact_location, exist_ok=True)
        experiment_id = mlflow.create_experiment(
            name,
            artifact_location=artifact_location,
            tags=tags,
        )
    if verbose:
        print(f"{load_or_create}l'expérience MLFlow '{name}', ID = {experiment_id}")
    return experiment_id


# Convertit une figure matplotlib en image pil pour pouvoir la loguer
# dans mlflow sans l'enregistrer sur disque auparavant (l'enregistre dans un buffer)
def pil_image_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def merge_dicts_with_prefixes(**named_dicts):
    """
    Prend des dictionnaires nommés et ajoute leur nom comme préfixe aux clés.

    Exemple :
    merge_dicts_with_prefixes(model=params1, metric=params2)
    => {"model__lr": 0.01, "metric__threshold": 0.5}
    """
    merged = {}
    for prefix, d in named_dicts.items():
        merged.update({f"{prefix}__{k}": v for k, v in d.items()})
    return merged


# Crée un dataset pour le loguer dans mlflow
# Pour conserver les SK_ID_CURR indexer X et y par SK_ID_CURR
def create_dataset(X, y, filepath):
    # Pour loguer dans mlflow, X et la target doivent figurer dans le même fichier
    pd_dataset = pd.concat([X, y], axis=1)

    # Par ailleurs il ne doit pas y avoir d'int avec des NaN
    # Le plus simple est de convertir tous les int en float
    int_columns = pd_dataset.select_dtypes(include="int").columns.to_list()
    pd_dataset[int_columns] = pd_dataset[int_columns].astype(np.float16)

    # Créer le fichier csv maintenant ?

    dataset = mlflow.data.from_pandas(
        pd_dataset,  # DataFrame pandas avec les int castés en float
        name=filepath,  # Path complet du fichier CSV
    )
    return dataset


def log_scoring_params(
    bg_params, threshold_prob, artifact_path="scoring_params", filename="scoring.json"
):
    """
    À partir des paramètres utilisés pour la métrique métier et du seuil de probabilité utilisé, loggue un dictionnaire dans MLflow
    comme artifact JSON sans écrire sur disque.

    Args:
        bg_params: (dict) Les paramètres utilisés pour la métrique métier.
        treshold_prob: (float) Le seuil de probabilité utilisé pour scorer le modèle
        artifact_path (str): Dossier dans les artifacts MLflow. (c'est un sous dossier dans l'interface, ce n'est pas le chemin physique)
        filename (str): Nom du fichier JSON loggué.

    Returns:
        dict: Le dictionnaire loggué.
    """
    # 1. Ajoute le seuil de proba au dictionnaire de paramètres utilisés pour la métrique métier
    dic_scoring = bg_params.copy()
    dic_scoring["threshold_prob"] = threshold_prob

    # 2. Convertir en JSON en mémoire
    json_buffer = io.StringIO()
    json.dump(dic_scoring, json_buffer, indent=2)
    json_buffer.seek(0)

    # 3. Logger dans MLflow
    mlflow.log_text(json_buffer.read(), artifact_file=f"{artifact_path}/{filename}")
    return dic_scoring


def log_image(fig, fig_name, artifact_path="plots"):
    img = pil_image_from_fig(fig)
    artifact_file = os.path.join(artifact_path, f"{fig_name}.png")
    mlflow.log_image(img, artifact_file=artifact_file)


def log_features(X, artifact_path="features", filename="features.json"):
    """
    À partir d'un DataFrame X, loggue un dictionnaire {feature: dtype} dans MLflow
    comme artifact JSON sans écrire sur disque.

    Args:
        X (pd.DataFrame): Le DataFrame des features.
        artifact_path (str): Dossier dans les artifacts MLflow. (c'est un sous dossier dans l'interface, ce n'est pas le chemin physique)
        filename (str): Nom du fichier JSON loggué.

    Returns:
        dict: Le dictionnaire {feature: dtype} loggué.
    """
    # 1. Construire le dict {feature: type}
    feature_dic = {col: str(dtype) for col, dtype in X.dtypes.items()}

    # 2. Convertir en JSON en mémoire
    json_buffer = io.StringIO()
    json.dump(feature_dic, json_buffer, indent=2)
    json_buffer.seek(0)

    # 3. Logger dans MLflow
    mlflow.log_text(json_buffer.read(), artifact_file=f"{artifact_path}/{filename}")
    return feature_dic


# Supprime toutes les expériences mlflow qui commencent par un prefixe (dans l'interfacd web et sur le disque)
def delete_experiments_with_prefix(prefix, artifacts_dir="mlflow_artifacts"):
    client = MlflowClient()
    # experiments = client.list_experiments()
    experiments = mlflow.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )

    experiments_to_delete = [
        experiment for experiment in experiments if experiment.name.startswith(prefix)
    ]

    experiment_ids_to_delete = [
        experiment.experiment_id for experiment in experiments_to_delete
    ]

    experiment_names_to_delete = [
        experiment.name for experiment in experiments_to_delete
    ]
    print(f"{len(experiment_names_to_delete)} expériences à supprimer dans mlflow :")
    print(experiment_names_to_delete)

    for id in experiment_ids_to_delete:
        client.delete_experiment(id)

    # On supprime également les artifacts sauvegardés sur disque correspondant aux exépriences
    for name in experiment_names_to_delete:
        dir_name = os.path.join(artifacts_dir, name)
        shutil.rmtree(dir_name)

    return experiment_names_to_delete
