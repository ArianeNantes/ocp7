import numpy as np
import random
import logging
import re
import requests
import os
import gc
import time
import joblib
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import psycopg2
from psycopg2 import OperationalError

import inspect
import lightgbm as lgb
import optuna
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage
from scipy.stats import beta
import plotly
import kaleido
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
)
import cudf
from cuml.linear_model import LogisticRegression

from src.p7_constantes import NUM_THREADS
from src.p7_util import timer, clean_ram, format_time
from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_constantes import MODEL_DIR, DATA_INTERIM, MAX_SIZE_PARA, VAL_SEED

from src.p7_secret import (
    PORT_MLFLOW,
    HOST_MLFLOW,
    PORT_PG,
    HOST_PG,
    PASSWORD_PG,
    USER_PG,
)

from src.p7_simple_kernel import (
    # get_batch_size,
    get_memory_consumed,
    get_available_memory,
)
from src.p7_metric import pred_prob_to_binary, penalize_f1, penalize_business_gain
from src.p7_evaluate import cuml_cross_validate


class ExperimentSearch:
    def __init__(
        self,
        metric="auc",
        direction="maximize",
    ):
        self.input_dir = DATA_INTERIM
        self.train_filename = "00_v3_train.csv"
        self.predictors_name = "features_sorted_by_importance.pkl"
        # Par défaut on ne réduit pas les colonnes
        self.n_predictors = None
        # Par défaut on ne réduit pas les lignes
        self.frac_sample = 1.0
        self.output_dir = MODEL_DIR
        self.task = "search"
        self.model_name = "debug"
        self.pipe_preprocess = None
        self.sampling = None
        self.objective_name = "binary"
        self.metric = metric
        self.dic_metrics = {"auc"}
        self.direction = direction
        self.sampler = optuna.samplers.TPESampler()
        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource=400, reduction_factor=3
        )
        self.storage = RDBStorage(
            f"postgresql://{USER_PG}:{PASSWORD_PG}@localhost:{PORT_PG}/optuna_db"
        )
        self.num = 0
        self.continue_if_exist = False
        self.optimize_boosting_type = True
        self.uri_mlflow = f"http://{HOST_MLFLOW}:{PORT_MLFLOW}"
        # Attributs à construire obligatoirement avec initialisation

        self.name = ""
        self.study_name = ""
        self.description = ""
        self.tags = {}
        self.X = None
        self.y = None
        self.random_state = VAL_SEED
        self.mlflow_id = None
        self.study = None
        self.initialized = False
        self.parent_run_id = None
        self.device = "cuda"
        # Turn off optuna log notes.
        # optuna.logging.set_verbosity(optuna.logging.WARN)

    def check_postgresql_server(
        self,
        host="localhost",
        port=PORT_PG,
        dbname="optuna_db",
        user=USER_PG,
        password=PASSWORD_PG,
    ):
        try:
            connection = psycopg2.connect(
                host=host, port=port, dbname=dbname, user=user, password=password
            )
            connection.close()
            print("La connexion Postgresql est OK")
        except OperationalError as e:
            self.initialized = False
            print(
                "La connexion PostgreSQL a échoué. Vérifier que le service est démarré"
            )
            print(
                "Pour démarrer le service, vous pouvez utiliser la méthode experiment.start_postresql_server()"
            )
            print("Error:", e)
        """
        On peut vérifier le statut du serveice en ligne de commande :
        sudo service postgresql status (mdp depuis secret)
        :q pour sortir
        Redémarrer :
        sudo service postgresql restart
        """

    def check_mlflow_server(self):
        # Use the fluent API to set the tracking uri and the active experiment
        mlflow.set_tracking_uri(self.uri_mlflow)
        try:
            response = requests.get(self.uri_mlflow)
            if response.status_code == 200:
                print("Le serveur MLFlow est Ok")
            else:
                print(
                    "La connexion MLFlow a échoué. Status code:",
                    response.status_code,
                )
        except requests.exceptions.RequestException as e:
            self.initialized = False
            print("La connexion MLFlow a échoué. Vérifiez que le serveur est démarré.")
            print(
                "Pour le démarrer, vous pouver utiliser experiment.start_mlflow_server()"
            )
            print("Error:", e)

    def check_services(self):
        """print(
            "# [TODO] Vérifier que les services sont démarrés (serveur mlflow et postgresql)"
            #Démarrer un serveur mlflow local en ligne de commande :
            #mlflow server --host 127.0.0.1 --port 8080

        )"""

        # Use the fluent API to set the tracking uri and the active experiment
        mlflow.set_tracking_uri(self.uri_mlflow)
        return

    def start_mlflow_server(
        self, host=HOST_MLFLOW, port=PORT_MLFLOW, check_connect=True
    ):
        cmd = [
            "mlflow",
            "server",
            # "--backend-store-uri", backend_store_uri,
            # "--default-artifact-root", default_artifact_root,
            "--host",
            host,
            "--port",
            port,
        ]
        env = os.environ.copy()

        # Start the MLflow server
        print("Démarrage du serveur MLFlow")
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Allow some time for the server to start
        time.sleep(5)
        if check_connect:
            self.check_mlflow_server()
        return process

    def check_directories(self):
        """print(
            "# [TODO] Vérifier que les directories sont ok et les créer si nécessaire"
        )"""
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            print(f"Répertoire {self.output_dir} créé")
        return

    def init_name(
        self,
        model_name=None,
        objective_name=None,
        num=None,
        verbose=True,
        continue_if_exist=False,
    ):
        if model_name:
            self.model_name = model_name
        if objective_name:
            self.objective_name = objective_name
        if num:
            self.num = num
        if continue_if_exist is not None:
            self.continue_if_exist = continue_if_exist

        root_name = f"{self.task}_{self.model_name}_{self.objective_name}"
        name = root_name + f"_{self.num:03}"

        # On vérifie si l'expérience existe déjà dans mlflow
        existing_experiment = mlflow.get_experiment_by_name(name)

        # Si on autorise le chargement d'une expérience qui existe déjà
        detail = " (nouvelle)"
        if self.continue_if_exist:
            if existing_experiment:
                detail = " déjà existante"

        # Si on n'autorise pas un nom d'expérience qui existe déjà, on augmente le numéro jusqu'à une qui n'existe pas
        else:
            while existing_experiment:
                self.num += 1
                num_str = f"_{self.num:03}"
                name = root_name + num_str
                existing_experiment = mlflow.get_experiment_by_name(name)
                detail = " (nouvelle - numéro augmenté)"
            self.name = name

        if verbose:
            print(f"Nom de l'expérience{detail} : {self.name}")

        return name, existing_experiment

    # Appeler cette fonc après init_name
    def init_description(
        self,
        objective_name=None,
        metric=None,
        add_description="",
        add_tags={},
        verbose=True,
    ):
        if objective_name:
            self.objective_name = objective_name
        if metric:
            self.metric = metric

        main_description = (
            f"Recherche bayesienne d'hyperparamètres pour modèle {self.model_name}"
        )
        main_tags = {
            "task": self.task,
            "model": self.model_name,
            "objective": self.objective_name,
            "metric": self.metric,
            "num": f"{self.num:03}",
            "db": "optuna",
        }
        description = main_description
        tags = main_tags
        if add_description:
            description = main_description + "\n" + add_description
        if add_tags:
            for k, v in add_tags.items():
                tags[k] = v

        self.description = description
        self.tags = tags

        if verbose:
            print(f"Description de '{self.name}' :")
            print(self.description)
            print("Tags :")
            for k, v in self.tags.items():
                print(f"\t'{k}' : {v}")
        return self.description, self.tags

    # Lecture des données AVANT resampling de type smote etc.
    def init_data(
        self,
        frac_sample=None,
        input_dir=None,
        train_filename=None,
        predictors_name=None,
        n_predictors=None,
        verbose=True,
    ):
        if frac_sample:
            self.frac_sample = frac_sample
        if input_dir:
            self.input_dir = input_dir
        if train_filename:
            self.train_filename = train_filename
        if predictors_name:
            self.predictors_name = predictors_name
        if n_predictors:
            self.n_predictors = n_predictors
        if frac_sample:
            self.frac_sample = float(frac_sample)
            # [TODO] Pas tags mais fonc add_tags qui ajoute des tags à une expérience déjà créée ou initialisée
            self.tags[frac_sample] = f"{frac_sample:0%}"

        ##################### Lecture de toutes les données de train (=avec target ET déjà partagé pour laisser un test.csv)
        if verbose:
            print(
                f"Chargement des données. n_predictors={self.n_predictors}, frac_sample={self.frac_sample}"
            )

        all_train = cudf.read_csv(os.path.join(self.input_dir, self.train_filename))
        if self.device != "cuda":
            all_train = all_train.to_pandas()
            # S'il y a des nan dans des bool ça donnera des object, on les convertit en float
            object_features = all_train.select_dtypes("object").columns
            all_train[object_features] = all_train[object_features].astype("float64")

        to_drop = sel_var(all_train.columns, "Unnamed", verbose=False)
        if to_drop:
            all_train = all_train.drop(to_drop, axis=1)
        all_train = all_train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))

        ##################### Réduction du nombre de colonnes
        reduction_predictor = f"Tous les prédicteurs, "
        # On lit les predicteurs du Kernel simple triés par ordre d'importance
        # [TODO] modifier le principe de réduction de lignes et de colonnes
        # all_predictors = joblib.load(os.path.join(self.input_dir, self.predictors_name))
        all_predictors = [f for f in all_train if f not in ["TARGET"]]

        # Si le nombre de prédicteurs n'est pas précisé (None) ou s'il dépasse, on les prends tous
        # sinon on réduira le nombre de colonnes
        if not self.n_predictors or self.n_predictors > len(all_predictors):
            self.n_predictors = len(all_predictors)
        if len(all_predictors) > self.n_predictors:
            reduction_predictor = "Après réduction des prédicteurs, "

        X, y = (
            all_train[all_predictors[: self.n_predictors]],
            all_train["TARGET"],
        )
        self.X = X
        self.y = y

        # [TODO] mettre device en paramètre de la config et prévoir le cas pandas
        if verbose:
            print(
                "Forme initiale de X :",
                all_train[all_predictors].shape,
                "type :",
                type(self.X),
            )
            print(
                f"{reduction_predictor}forme de X : {self.X.shape}, conso mémoire : {get_memory_consumed(self.X, verbose=False):.0f} Mo"
            )
        del all_train, all_predictors
        gc.collect()

        ##################### Réduction du nombre de lignes (avant under_sampling)
        row_reduction = "Toutes les lignes, "
        if self.frac_sample < 1.0:
            X_sampled, _, y_sampled, _ = train_test_split(
                X,
                y,
                train_size=self.frac_sample,
                random_state=self.random_state,
                stratify=y,
            )
            self.X = X_sampled
            self.y = y_sampled
            row_reduction = (
                f"Après réduction des lignes (échantillon {self.frac_sample:.0%}), "
            )
        if verbose:
            memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
            print(
                f"{row_reduction}forme de X : {self.X.shape}, conso mémoire {memory_consumed_mb:.0f} Mo"
            )

        ##################### Warning taille memoire
        mem_consumed = get_memory_consumed(self.X, verbose=False)
        if mem_consumed > MAX_SIZE_PARA:
            print(
                f"WARNING: X est trop gros ({mem_consumed:.0f} > {MAX_SIZE_PARA} Mo) pour une réelle parallélisation des threads"
            )
        return self.X, self.y

    # Appeler init_data avant cette fonc
    def init_resampling(self):
        # [TODO] Implémenter le resampling ex smote
        return

    # Appeler init_expriment_name avant cette fonc et check service
    def create_or_load_experiment(self, verbose=True):

        tags = self.tags
        tags["mlflow.note.content"] = self.description

        # On vérifie si l'expérience existe déjà
        existing_experiment = mlflow.get_experiment_by_name(self.name)

        load_or_create = "Création de "
        if existing_experiment:
            experiment_id = existing_experiment.experiment_id
            load_or_create = "Chargement de "
        else:
            experiment_id = mlflow.create_experiment(
                self.name,
                artifact_location=self.output_dir,
                tags=tags,
            )
        self.mlflow_id = experiment_id
        if verbose:
            print(
                f"{load_or_create}l'expérience MLFlow '{self.name}', ID = {self.mlflow_id}"
            )
        # self.create_parent_run()
        return self.mlflow_id

    # Appeler init_name avant cette fonc, et check_service (pg doit être démarré)
    def create_or_load_study(
        self,
        objective_name=None,
        direction=None,
        storage=None,
        sampler=None,
        pruner=None,
    ):
        if objective_name:
            self.objective_name = objective_name
        if direction:
            self.direction = direction
        if storage:
            self.storage = storage
        if sampler:
            self.sampler = sampler
        if pruner:
            self.pruner = pruner

        study = optuna.create_study(
            study_name=self.name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=True,
        )
        self.study = study
        return self.study

    def init_config(self, verbose=True, **kwargs):
        # La liste des paramètres possibles à passer dans kwargs sont
        # les attributs qui ne sont pas protégés ni privés et qui ne sont pas des méthodes
        public_attributes = [
            a
            for a in dir(self)
            if not a.startswith("_")
            and not a.startswith("__")
            and not callable(getattr(self, a))
        ]

        valid_args = [arg for arg in kwargs.keys() if arg in public_attributes]
        invalid_args = [arg for arg in kwargs.keys() if arg not in valid_args]

        # On met à jour les paramètres qui sont autorisés
        for k in valid_args:
            v = kwargs[k]
            self.__setattr__(k, v)

        if verbose:
            print(f"Les paramètres {valid_args} ont été mis à jour")
        if invalid_args:
            print(
                f"Les paramètres {invalid_args} ne sont pas valides. Liste des paramètres possibles :"
            )
            # [TODO] Trier par ordre alhabétique ?
            print(public_attributes)

        # On ne veut pas ré-initialiser les data pour rien (la lecture est longue)
        # donc on récupère les arguments de kwargs qui sont aussi dans la signature de init_data
        # on n'exécute init_data que si les df X ou y sont vides ou si des arguments de data ont été passés dans kwargs, même s'ils ne sont pas différents d'avant
        data_signature = inspect.signature(self.init_data)
        data_args = [param.name for param in data_signature.parameters.values()]
        data_args_to_init = [a for a in kwargs.keys() if k in data_args]
        if self.X is None or self.y is None or data_args_to_init:
            self.init_data()
        self.init_name()
        self.init_description()
        self.check_directories()
        self.check_services()

        if invalid_args:
            return False
        else:
            self.initialized = True
            return self.initialized

    # Appeler les init avant les create
    def create_or_load(self):
        id_experiment = self.create_or_load_experiment()
        study = self.create_or_load_study()
        return id_experiment, study

    def create_parent_run(self):
        # On crée un run parent qui va contenir tous les runs (un run enfant par trial)
        parent_run = mlflow.start_run(
            experiment_id=self.mlflow_id,
            run_name="Study",
            tags=self.tags,
            # description="Résultats de la recherche d'hyperparamètres",
        )
        self.parent_run_id = parent_run.info.run_id
        mlflow.end_run()

    def create_best_run(self):
        with mlflow.start_run(
            experiment_id=self.mlflow_id,
            run_name=f"T_{self.study.best_trial.number}_best_of_{len(self.study.get_trials())}_trials",
        ):
            mlflow.log_params(self.study.best_trial.params)
            mlflow.log_metric(self.metric, self.study.best_trial.value)
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.write_html(os.path.join(self.output_dir, "hyperparam_importance.html"))
            mlflow.log_artifact(
                os.path.join(self.output_dir, "hyperparam_importance.html")
            )
            """fig = optuna.visualization.plot_parallel_coordinate(
                self.study,
                # params=["boosting_type", "n_estimators", "num_leaves", "learning_rate"],
            )
            fig.write_html(os.path.join(self.output_dir, "parallel_coordinate.html"))
            mlflow.log_artifact(
                os.path.join(self.output_dir, "parallel_coordinate.html")
            )"""
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            mlflow.log_artifact(
                os.path.join(self.output_dir, "optimization_history.html")
            )

    # Inutile, on va plutôt faire un run parent
    def track_best_run(self):
        # find the best run, log its metrics as the final metrics of this run.
        # client = MlflowClient()

        # Récupération de tous les runs de l'experiment
        runs = mlflow.search_runs(experiment_ids=[self.mlflow_id])
        print("len(runs)", len(runs), "type", type(runs))
        # print(runs)

        # On enlève le parent
        # nested_runs = [run for run in runs if run.run_name != "best"]
        # print(runs.columns)
        nested_runs = runs[runs["tags.mlflow.parentRunId"].notnull()].sort_values(
            by=f"metrics.{self.metric}", ascending=False
        )
        best_run = nested_runs.head(1)
        print("len(nested_runs)", nested_runs.shape)

        # parent_run_name = "best"
        """runs = client.search_runs(
            [self.mlflow_id], f"tags.mlflow.parentRunName == 'best'"
        )"""

        # Filtrer les runs qui ont le run parent désiré
        # nested_runs = runs[runs["tags.mlflow.parentRunName"] == parent_run_name]
        # nested_runs = [runs[runs["tags.mlflow.parentRunId"]] == self.parent_run_id]

        """best_metric = 0

        for r in nested_runs:
            if r.data.metrics[self.metric] > best_metric:
                best_run = r
        if best_run:
            print("Metrics du best", best_run.data.metrics)
        else:
            print("best run non trouvé")"""

        """mlflow.set_tag("best_run", best_run.info.run_id)
        mlflow.log_metrics(
            {
                f"train_{metric}": best_val_train,
                f"val_{metric}": best_val_valid,
                f"test_{metric}": best_val_test,
            }
        )"""
        return best_run

    # Permet de suggérer des float en en proposant plus autour de 0.5 si
    # Si les params alpha et beta sont égaux et low=0 et high=1
    def suggest_beta(self, param_name, low=0, high=1, alpha=2.0, beta_param=2.0):
        # Sample a value from the beta distribution
        value = beta.rvs(alpha, beta_param)
        # Scale the value to the desired range [low, high]
        scaled_value = low + (high - low) * value
        return scaled_value


class ExperimentSearchCuml(ExperimentSearch):
    def __init__(self, metric="auc", direction="maximize"):
        super().__init__(metric=metric, direction=direction)
        self.device = "cuda"
        self.model_name = "logreg"
        # Turn off optuna log notes.
        # optuna.logging.set_verbosity(optuna.logging.WARN)

    def objective(self, trial):

        with mlflow.start_run(
            experiment_id=self.mlflow_id, run_name=f"{trial.number}", nested=True
        ):
            # doc cuml : https://docs.rapids.ai/api/cuml/nightly/api/#regression-and-classification
            c = trial.suggest_float("C", 1e-7, 100.0, log=True)

            # En cuml, pas le choix du solver mais
            # si penalty=None ou l2 solver=L-BFGS,
            # si penalty=l1 ou elasticnet avec un ratio_l1 > 0 alors solver=OWL-QN

            penalty = trial.suggest_categorical(
                "penalty", ["none", "l2", "l1", "elasticnet"]
            )
            # penalty = trial.suggest_categorical("penalty", ["l1", "elasticnet"])
            if penalty == "elasticnet":
                l1_ratio = trial.suggest_float("l1_ratio", 0.2, 0.8, log=False)
            else:
                l1_ratio = None

            class_weight = trial.suggest_categorical(
                "class_weight", ["none", "balanced"]
            )
            # class_weight = "balanced"
            # Pour avoir l'équivallent d'un tol sklearn en cuml, il faut diviser le tol sklearn par sample_size
            # tol = trial.suggest_float("tol", 1e-6, 1e-3)
            fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
            # fit_intercept = False
            clf_params = {
                "C": c,
                "penalty": penalty,
                "l1_ratio": l1_ratio,
                "fit_intercept": fit_intercept,
            }
            # Petit bug pour le cuml LogisticRegression, on ne peut pas directement assigner class_weight
            if class_weight == "balanced":
                clf_params["class_weight"] = "balanced"

            # Max_iter par défaut est 1_000. Si une régularisation l1 est appliquée, dans ces conditions,
            # L'algorithme ne pourra pas converger. Soit on augmente max_iter, soit on scale les features binaires dans le pipe.
            if penalty == "l1" or penalty == "elasticnet":
                clf_params["max_iter"] = 10_000
                if penalty == "elasticnet":
                    clf_params["l1_ratio"] = l1_ratio

            # On concentre la recherche du seuil autour de 0.5 avec la distribution Beta
            # threshold_prob = self.suggest_beta("threshold_prob", low=0.05, high=0.95)
            threshold_prob = trial.suggest_float(
                "threshold_prob", 0.05, 0.95, log=False
            )
            print(
                "Trial",
                trial.number,
                "penalty :",
                penalty,
                "l1_ratio :",
                l1_ratio,
                "threshold_prob :",
                threshold_prob,
            )

            clf = LogisticRegression(**clf_params)
            scores = cuml_cross_validate(
                self.X,
                self.y,
                self.pipe_preprocess,
                clf,
                threshold_prob=threshold_prob,
            )

            mean_scores = scores.mean(axis=0)

            # Minimise la log loss : ll = log_loss(test_y , y_predlr)

            # Fin d'un trial
            # del folds
            # gc.collect()
            dic_metrics = mean_scores.to_dict()
            print(
                "Trial",
                trial.number,
                "dic_metrics",
                dic_metrics,
                "type",
                type(dic_metrics),
            )

            # Pour logguer les métriques additionnelles dans optuna_db :
            # trial.set_user_attr("constraint", [c0])

            """for k, v in dic_metrics.items():
                mlflow.log_metric(k, v)"""
            mlflow.log_metrics(dic_metrics)
            all_params = {
                "C": c,
                "penalty": penalty,
                # "l1_ratio": l1_ratio,
                "fit_intercept": fit_intercept,
                "class_weight": class_weight,
                "threshold_prob": threshold_prob,
            }
            mlflow.log_params(all_params)
        return dic_metrics[self.metric]

    def run_logreg_trials(self, n_trials=20, n_jobs=1, verbose=True):
        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials...")

        self.study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )
        if verbose:
            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))

    # La parallélisation ne fonctionne pas si CUDA car pas assez de mémoire
    # ou tout simplement allocation sur CUDA en parallel ne fonctionne pas bien (alloue plus que nécessaire).
    # On peut surveiller en temps réel la consommation de VRAM avec nvidia-smi -l 1 (crée une loop pour rafraichir toutes les 1 secondes)
    # Le multithreads réglerait peut-être le problème de l'allocation VRAM mais MLFlow n'est pas thread-safe.
    def optimize(self, n_trials=10, verbose=True):
        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials sur CUDA...")

        # self.run_logreg_trials(n_trials=n_trials, n_jobs=1, verbose=False)
        self.study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )

        if verbose:
            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))
        self.create_best_run()


class ExperimentSearchLgbm(ExperimentSearch):
    def __init__(self, metric="auc", direction="maximize"):
        super().__init__(metric=metric, direction=direction)
        self.device = "cpu"
        self.model_name = "lgbm"
        # Turn off optuna log notes.
        # optuna.logging.set_verbosity(optuna.logging.WARN)

    def objective(self, trial):
        # parent_run_name = "best"
        # Attention les n° de trial ne correspondent pas à ceux fournis en logging stdout
        with mlflow.start_run(
            experiment_id=self.mlflow_id, run_name=f"{trial.number}", nested=True
        ):
            boosting_type = trial.suggest_categorical("boosting_type", ["dart", "gbdt"])
            lambda_l1 = (trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),)
            lambda_l2 = (trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),)
            num_leaves = (trial.suggest_int("num_leaves", 8, 256),)
            feature_fraction = (trial.suggest_float("feature_fraction", 0.4, 1.0),)
            bagging_fraction = (trial.suggest_float("bagging_fraction", 0.4, 1.0),)
            bagging_freq = (trial.suggest_int("bagging_freq", 1, 7),)
            min_child_samples = (trial.suggest_int("min_child_samples", 5, 100),)
            learning_rate = (
                trial.suggest_float("learning_rate", 0.0001, 0.5, log=True),
            )
            max_bin = trial.suggest_int("max_bin", 128, 512, step=32)
            n_estimators = trial.suggest_int("n_estimators", 40, 400, step=20)
            # On concentre la recherche du seuil autour de 0.5 avec la distribution Beta
            threshold_prob = self.suggest_beta("threshold_prob", low=0.1, high=0.9)

            hyperparams = {
                "boosting_type": boosting_type,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "min_child_samples": min_child_samples,
                "learning_rate": learning_rate,
                "max_bin": max_bin,
                "n_estimators": n_estimators,
                "threshold_prob": threshold_prob,
            }

            # 'binary' est la métrique d'erreur
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, self.objective_name
            )

            # Pour intégration optuna mlflow avec nested runs voir :
            # https://mlflow.org/docs/latest/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.html?highlight=run%20description

            # On n'utilise pas cross_val_score pour des problèmes de RAM
            # scores = cross_val_score(model, X, y, scoring="f1_macro", cv=5)
            folds = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=self.random_state
            )
            dic_val_scores = {
                "auc": [],
                "accuracy": [],
                "recall": [],
                "penalized_f1": [],
                "business_gain": [],
                "fit_time": [],
                "tn": [],
                "tp": [],
                "fn": [],
                "fp": [],
            }

            predictors = [
                f for f in self.X.columns if f not in ["SK_ID_CURR", "TARGET"]
            ]

            # Dans le split, la copie est indispensable car sinon on a une Vue array avec FLAG qui rend l'array immutable et
            # provoque l'erreur 'cannot set WRITEABLE flag to True of this array'.
            for _, (train_idx, valid_idx) in enumerate(
                folds.split(
                    # Pour construire les indices de folds, nous n'avons besoin pour X que de 2 variables afin d'obtenir un DataFrame,
                    # En choisir seulement 2 (si copie) accélère émormément les traitements
                    self.X[predictors[:2]].to_numpy(copy=True),
                    self.y.to_numpy(copy=True),
                )
            ):
                # Ici on transforme en array et on copie car cela accélère la parallélisation
                X_train = self.X.loc[train_idx, predictors].to_numpy(copy=True)
                y_train = self.y.loc[train_idx].to_numpy(copy=True)
                X_val = self.X.loc[valid_idx, predictors].to_numpy(copy=True)
                y_val = self.y.loc[valid_idx].to_numpy(copy=True)

                model = lgb.LGBMClassifier(
                    # force_row_wise=True,
                    force_col_wise=True,
                    objective=self.objective_name,
                    is_unbalanced=False,
                    boosting_type=boosting_type,
                    n_estimators=n_estimators,
                    lambda_l1=lambda_l1,
                    lambda_l2=lambda_l2,
                    num_leaves=num_leaves,
                    feature_fraction=feature_fraction,
                    bagging_fraction=bagging_fraction,
                    bagging_freq=bagging_freq,
                    min_child_samples=min_child_samples,
                    learning_rate=learning_rate,
                    max_bin=max_bin,
                    callbacks=[pruning_callback],
                    random_state=VAL_SEED,
                    verbose=-1,
                )
                t0_fit_time = time.time()
                model.fit(
                    X_train,
                    y_train,
                    # eval_set=[(X_train, y_train), (X_val, y_val)],
                    eval_set=(X_val, y_val),
                    eval_metric=self.metric,
                )
                fit_time = time.time() - t0_fit_time

                # y_score = np.array(model.predict_proba(X_val_np)[:, 1], copy=True)
                # La copie a tendance à augmenter la rapidité ? Bizarre
                y_score_val = model.predict_proba(X_val)[:, 1]
                # Prédiction de la classe en fonction du seuil de probabilité

                y_pred_val = pred_prob_to_binary(y_score_val, threshold=threshold_prob)

                # Mesures sur le jeu de validation
                dic_val_scores["auc"].append(roc_auc_score(y_val, y_score_val))
                dic_val_scores["accuracy"].append(accuracy_score(y_val, y_pred_val))
                dic_val_scores["recall"].append(recall_score(y_val, y_pred_val))
                dic_val_scores["penalized_f1"].append(penalize_f1(y_val, y_pred_val))
                dic_val_scores["business_gain"].append(
                    penalize_business_gain(y_val, y_pred_val)
                )
                dic_val_scores["fit_time"].append(fit_time)
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()
                dic_val_scores["tn"].append(tn)
                dic_val_scores["fn"].append(fn)
                dic_val_scores["tp"].append(tp)
                dic_val_scores["fp"].append(fp)

                del X_train
                del y_train
                del X_val
                del y_val
                del model
                gc.collect()

            # Fin d'un trial
            del folds
            gc.collect()
            dic_metrics = {k: np.mean(v) for (k, v) in dic_val_scores.items()}

            # Pour logguer les métriques additionnelles dans optuna_db :
            # trial.set_user_attr("constraint", [c0])

            """for k, v in dic_metrics.items():
                mlflow.log_metric(k, v)"""
            mlflow.log_metrics(dic_metrics)
            mlflow.log_params(hyperparams)
        return dic_metrics[self.metric]

    def run_lgb_trials(self, n_trials=10, reseed_sampler=False):
        if reseed_sampler:
            self.study.sampler = optuna.samplers.TPESampler(seed=VAL_SEED)
            self.study.sampler.reseed_rng()
        else:
            self.study.sampler = optuna.samplers.TPESampler()

        self.study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )

    # Pourquoi Lgbm sur CPU et pas cuda / gpu contrairement aux modèles linéaires ?
    # RAPID avec cudf ou cuml ne supporte pas lightgbm puisque lightgbm apporte son propre support GPU. Cependant,
    # Lgbm existe sur GPU mais n'est pas performant du tout si beaucoup de vraiables en OneHot.
    # Dans notre cas (beaucoup de OneHot), le traitement GPU est plus long que le le traitement CPU.
    # On choisit donc CPU pour ce modèle, mais on tente de paralléliser.

    # Parallélisation des jobs sur CPU :
    # La vraie parallélisation est beaucoup plus rapide que le multithread (multithread optuna = optimize avec n_jobs > 1)
    # share_sampler_seed permet de s'assurer que le sampler ne va pas proposer les mêmes combinaisons de paramètres dans chaque workers,
    # grâce à la méthode opuna .reseed_rng().
    # cependant, les traitements sont alors vraiment beaucoup plus longs
    # (comme le sampler doit être partagé entre les workers, il s'agit probbablement de multithrading).

    # Si share_sampler_seed est false, on applique tout simplement un sampler avec une graine différente dans chaque worker,
    # C'est BEAUCOUP PLUS RAPIDE (du simple au double au minimum).
    # Les résultats ne sont pas reproductibles en parallélisation quelque soit la valeur de share_sampler_seed (True ou False), Mais
    # les résultats sont un PETIT PEU MEILLEURS avec share_sampler_seed=True qu'avec share_sampler_seed=False .
    def optimize(
        self,
        n_workers=4,
        n_trials_per_worker=10,
        share_sampler_seed=False,
        verbose=True,
    ):
        t0 = time.time()

        if verbose:
            print(
                f"Optimisation {n_trials_per_worker * n_workers} trials sur CPU. Parallélisation : {n_workers} workers de {n_trials_per_worker} trials chacun..."
            )
        if share_sampler_seed:
            print("Seed recalculée pour chaque worker.")
            Parallel(n_jobs=n_workers)(
                delayed(self.run_lgb_trials)(n_trials_per_worker, True)
                for _ in range(n_workers)
            )
        else:
            print("Seed aléatoire pour chaque worker.")

            Parallel(n_jobs=n_workers)(
                delayed(self.run_lgb_trials)(n_trials_per_worker, False)
                for i in range(n_workers)
            )

        if verbose:
            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))
        self.create_best_run()


"""
La fonction _objectif est appelée une fois pour chaque essai (trial).
Ici on entraîne un LGBMClassfier et on calcule la métrique : ? replacer le f1-score par une autre
Optuna passe un objet trial à la fonction _objectif, qu'on peut utiliser pour en définir les paramètres.
log=True : applique une log scale aux valeurs à tester dans l'étendue spécifiée (pour les valeurs num), 
Effet : plus de valeurs sont testées à proximité de la borne basse et moins (logarithmiquement) vers
la borne haute
Convient particulièrement bien au learning rate : on veut se concentrer sur des valeurs + petites et 
augmenter exponentiellement le pas des valeurs à tester pour les plus grandes

Le pruning callbac est le mécanisme qui applique le pruning dynamique pendant l'entraînement du modèle LightGBM 
dans le cadre d'une étude Optuna. 
1 - Initialisation du callback : 
Crée le callback de pruning qui surveillera l'entraînement du modèle LightGBM 
et effectuera le pruning selon les instructions spécifiées par l'étude Optuna
2 - Évaluation périodique : 
Pendant l'entraînement de LightGBM, le callback est appelé périodiquement
pour évaluer les performances du modèle en fonction des critères de pruning définis par Optuna.
3 - Décision de pruning : 
Le callback utilise les informations fournies par Optuna, 
telles que les valeurs d'objectif de l'essai actuel et les valeurs d'objectif des essais précédents, 
pour décider si l'essai actuel doit être pruned en fonction de son efficacité par rapport à d'autres essais.
4 - Pruning : 
Pruning appliqué si décidé (arrête l'entraînement de ce modèle LightGBM)
5 - Réévaluation et ajustement :
Après le pruning d'un essai, l'entraînement peut continuer avec les essais restants, 
et le processus de pruning peut être répété à intervalles réguliers jusqu'à ce que l'étude soit terminée.


"""


"""
Pruning (="Elagage") = technique pour arrêter prématurément l'exécution (train + eval) si 
il est peu probable que l'essai conduise à une amélioreration significative des performances
(l'entraînement est stoppé si l'algo sous-performe)

**** HYPERBAND

Hyperband est une technique de pruning qui combine le pruning de type "Successive halving" et
d'autres techniques (random search, multi-bracket resource allocation strategy).

1 - Successive Halving (Demi-seuccessifs):
Au début de l'optimisation, Hyperband crée un grand nombre de configurations d'hyperparamètres
et entraîne chaque modèle pendant un petit nombre d'itérations (eochs). 
Ensuite, il évalue les performances de ces modèles et conserve uniquement les meilleurs. 
Il répète ce processus plusieurs fois, en doublant à chaque fois le nombre d'itérations, 
mais en réduisant de moitié le nombre de configurations conservées.

Ainsi les modèles prometteurs bénéficient de plus d'itérations pour converger vers de bonnes performances,
tandis que les modèles moins prometteurs sont élagués à chaque étape.

2 - Pruning basé sur la performance:
À chaque étape du Successive Halving, Hyperband évalue périodiquement les performances des modèles en formation
et décide de "pruner" les modèles moins prometteurs. 
Les modèles qui n'atteignent pas un certain seuil de performance sont arrêtés prématurément.

*** MULTI_BRACKET RESSOURCE ALLOCATION strategy (Allocation de ressource à plusieurs 'brackets' ou niveaux / échelon)

divise les ressources disponibles (telles que le nombre d'itérations, d'évaluations de modèles, etc.)
en plusieurs "brackets" ou "niveaux" et alloue ces ressources de manière dynamique en fonction de la performance
des modèles à chaque niveau.

1 - Multi-brackets
L'algorithme divise le budget total de ressources (par exemple, le nombre maximal d'itérations ou le temps total de calcul) en plusieurs "brackets" ou "échelons".
Chaque bracket représente une étape distincte de l'algorithme où différents ensembles d'hyperparamètres sont évalués.
Chaque bracket est caractérisé par un budget de ressources spécifique, qui peut être soit le nombre d'itérations, soit le temps de calcul. 
Les brackets initiaux ont un budget élevé tandis que les brackets ultérieurs ont des budgets plus bas.

2 - Resource allocation
Alloue dynamiquement les ressources aux config d'hyperparams
Dans chaque bracket, Hyperband commence par évaluer un grand nombre de configurations d'hyperparamètres pour un nombre restreint d'itérations. 
Ensuite, il élimine les configurations sous-performantes et alloue davantage de ressources aux configurations les plus prometteuses.

Within each bracket, successive halving is applied to iteratively eliminate underperforming configurations and
allocate more resources to the remaining promising ones.
At the begining of each bracket, a new set of hyperparam configurations is sampled using random search.

=> reduce risk of missing good config
more efficient and effective hyperparam tuning.
"""


"""
SAMPLER TPE :
Tree-structured Parzen Estimator
TPESampler utilise une approche bayésienne pour estimer les distributions de probabilité des valeurs d'hyperparamètres,
en se basant sur les performances des essais précédents, 
afin de guider efficacement la recherche vers des régions prometteuses de l'espace des hyperparamètres.

1 - Initialisation des distributions :
L'algorithme TPE commence par définir des distributions de probabilité pour chaque hyperparamètre à optimiser.
Ces distributions peuvent être continues (comme des distributions gaussiennes) ou discrètes (comme des distributions uniformes).
Pour chaque hyperparamètre, deux distributions sont définies : 
une pour les valeurs considérées comme "bonnes" (positive), et une pour les valeurs considérées comme "mauvaises" (négative).

2 - Évaluation des essais :
À chaque étape de l'optimisation, 
TPE utilise les essais précédents pour estimer les distributions de probabilité des valeurs d'hyperparamètres.
Les essais sont divisés en deux groupes : 
ceux qui ont produit de bonnes performances et ceux qui ont produit de mauvaises performances, 
en fonction du critère d'objectif (par exemple, la précision d'un modèle).
Les distributions de probabilité sont mises à jour en utilisant les essais des deux groupes,
en ajustant les paramètres des distributions pour mieux modéliser les valeurs d'hyperparamètres qui ont conduit à de bonnes performances.

3 - Échantillonnage des essais :
Une fois que les distributions de probabilité ont été mises à jour, 
TPE échantillonne de nouvelles valeurs d'hyperparamètres à partir de ces distributions. 
Les valeurs échantillonnées sont généralement celles qui maximisent l'espérance d'une fonction d'acquisition 
(par exemple, l'espérance de l'amélioration de l'objectif) 
ou qui minimisent une fonction de coût (par exemple, l'espérance de la perte de l'objectif).

4 - Évaluation des performances :
Les nouvelles valeurs d'hyperparamètres échantillonnées sont utilisées pour effectuer de nouveaux essais.
Les performances de ces essais sont évaluées à l'aide de la fonction d'objectif, 
et les résultats sont utilisés pour mettre à jour les distributions de probabilité et répéter le processus.
"""

"""
Métriques pour notre cas :

Le Recall mesure la proportion de vrais positifs (défauts correctement prédits)
parmi tous les vrais positifs et faux négatifs (défauts réels).
En choisissant le Recall comme métrique, on se concentre sur la capacité du modèle à capturer la majorité des cas de défauts, 
minimisant ainsi le risque de faux négatifs (oublier de prédire un défaut lorsqu'il existe réellement).

Le F1-score est la moyenne harmonique de la précision et du recall.
Le F1-score est particulièrement utile lorsque les classes sont déséquilibrées car il donne plus de poids aux classes minoritaires. 
Il peut être une bonne métrique lorsqu'on veut trouver un compromis entre la précision et le recall.

AUC-ROC (Area Under the Receiver Operating Characteristic Curve) : L'AUC-ROC est une métrique qui mesure la capacité du modèle à classer
correctement les exemples positifs et négatifs. 
Elle serait robuste aux déséquilibres de classe et donne une indication de la capacité du modèle à discriminer entre les classes.
"""

"""
balanced accuracy
The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. 
It is defined as the average of recall obtained on each class.
The best value is 1 and the worst value is 0 when adjusted=False.
"""
