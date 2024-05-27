import numpy as np
import os
import gc
import time
import joblib

import psycopg2
from psycopg2 import OperationalError


import optuna
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pandas as pd

from src.p7_util import timer, clean_ram, format_time
from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_constantes import MODEL_DIR, DATA_INTERIM, MAX_SIZE_PARA
from src.p7_constantes import (
    LOCAL_HOST,
    PORT_MLFLOW,
    PORT_POSTGRE,
    PASSWORD_POSTGRE,
    USER_POSTGRE,
)


class DbOptuna:

    # Configuration des connexions à la base de données
    optuna_db_uri = (
        f"postgresql://{USER_POSTGRE}:{PASSWORD_POSTGRE}@localhost/optuna_db"
    )
    # mlflow_tracking_uri = 'postgresql://username:password@localhost/mlflow_db'
    mlflow_tracking_uri = f"{LOCAL_HOST}:{PORT_MLFLOW}"

    def __init__(self) -> None:

        # .set_tracking_uri(mlflow_tracking_uri)
        pass

    @classmethod
    def del_studies_from_mlflow(cls):
        # Dans MLFlow, les expériences de recherche d'hyperparamètres Optuna sont tagguées par le mot search
        mlflow_study_tag = "search"

        # On configure les URI de connexion
        optuna_db_uri = cls.optuna_db_uri
        # mlflow_tracking_uri = 'postgresql://username:password@localhost/mlflow_db'
        mlflow_tracking_uri = cls.mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        client = MlflowClient()

        # On liste toutes les expériences MLflow
        tagged_experiments = mlflow.search_experiments(
            view_type=ViewType.ACTIVE_ONLY, filter_string="tags.task = 'search'"
        )
        tagged_experiments = mlflow.search_experiments(
            view_type=ViewType.ACTIVE_ONLY, filter_string="name ILIKE 'light%'"
        )

        # "attributes.name LIKE 'x%' AND tags.group = 'y'"
        # for
        """for exp in experiments:
            experiment_id = exp.experiment_id
            # Seules les experiments taggée avec le tag de recherche sont concernées
            if mlflow_study_tag in exp.tags:
                print(f"Deleting Optuna study for MLflow experiment: {exp.name} (ID: {experiment_id})")
                # Charger l'étude Optuna associée en utilisant le nom de l'expérience MLflow
                try:
                    study = optuna.load_study(study_name=exp.name, storage=optuna_db_uri)
                    # Supprimer l'étude Optuna
                    optuna.delete_study(study_name=exp.name, storage=optuna_db_uri)
                    print(f"Optuna study '{exp.name}' deleted.")
                except KeyError:
                    print(f"No Optuna study found for MLflow experiment '{exp.name}'")"""
        print("Liste des studys dans mlflow :", tagged_experiments)
        return
