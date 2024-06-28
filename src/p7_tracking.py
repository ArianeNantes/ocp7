import numpy as np
import os
import gc
import time
import joblib
import subprocess
import inspect

import psycopg2
from psycopg2 import OperationalError
import requests

import optuna
from optuna.storages import JournalStorage, JournalFileStorage, RDBStorage

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

import pandas as pd

# import sklearn
# import cuml
import cudf

from src.p7_util import timer, clean_ram, format_time
from src.p7_file import make_dir
from src.p7_regex import sel_var
from src.p7_constantes import MODEL_DIR, DATA_INTERIM, VAL_SEED, MAX_SIZE_PARA
from src.p7_secret import (
    HOST_MLFLOW,
    PORT_MLFLOW,
    PORT_PG,
    PASSWORD_PG,
    USER_PG,
)

from src.p7_preprocess import train_test_split_nan
from src.p7_simple_kernel import get_memory_consumed


# Classe pour fixer les méta données d'une expérience MLFlow :
# Le nom de l'expérience, les tags et la description
class ExpMetaData:
    def __init__(self, task, model_name, num=0, debug=False):
        self.num = num
        self.task = task
        self.model_name = model_name
        self.suffix1 = ""
        self.suffix2 = ""
        self.debug = debug
        self.continue_if_exist = False
        self._name = None
        self.main_description = ""
        self.extra_description = ""
        self._description = ""
        self._tags = {}
        self.main_tags = {
            "task": self.task,
            "model": self.model_name,
            "debug": self.debug,
        }
        self.extra_tags = {}

    @property
    def valid_args(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def _init_name(
        self,
        verbose=True,
    ):
        # Pour pouvoir réinitialiser plusieurs fois en repartant d'un numéro d'expérience 0,
        # on remet à zéro le num. Mais on laisse les autres params,
        # donc s'ils ne sont pas souhaités, il faut les supprimer dans les arguments, ex : suffix1="" ou extra_tags={}
        # Si ces params sont à None, ils ne seront pas modifiés par rapport à la fois d'avant (=non effacés)
        self.num = 0

        # En mode debug, la taâche est préfixée par debug
        if self.debug:
            root_name = f"debug_{self.task}_{self.model_name}"
        else:
            root_name = f"{self.task}_{self.model_name}"
        if self.suffix1:
            root_name = root_name + f"_{self.suffix1}"
        if self.suffix2:
            root_name = root_name + f"_{self.suffix2}"

        name = root_name + f"_{self.num:03}"

        # On vérifie si l'expérience existe déjà dans mlflow
        existing_experiment = mlflow.get_experiment_by_name(name)

        # Si on autorise le chargement d'une expérience qui existe déjà
        # Attention, une expérience qui n'existe pas dans MLFlow peut exister en tant que study dans la base optuna de PostgreSql
        detail = " créée (nouvelle dans MLFlow)"
        if self.continue_if_exist:
            if existing_experiment:
                detail = "chargée (déjà existante dans MLFlow)"

        # Si on n'autorise pas un nom d'expérience qui existe déjà, on augmente le numéro jusqu'à une qui n'existe pas
        else:
            while existing_experiment:
                self.num += 1
                num_str = f"_{self.num:03}"
                name = root_name + num_str
                existing_experiment = mlflow.get_experiment_by_name(name)
                detail = " (nouvelle - numéro augmenté)"
            self._name = name

        if verbose:
            print(f"Nom de l'expérience{detail} : {self._name}")
        return name, existing_experiment

    # Appeler cette fonc après init_name, pour changer les tags ou la description, appeler init
    def _init_tags(
        self,
        verbose=True,
    ):
        # Correspondance task et et description
        dic_description = {
            "search": "Recherche d'hyperparamètres",
            "permutation": "Suppression de features par permutation importance",
        }
        if self.task in dic_description.keys():
            task_description = dic_description[self.task]
        else:
            task_description = ""
        self.main_description = f"Modèle {self.model_name} - " + task_description

        description = self.main_description

        self.main_tags = {
            "task": self.task,
            "model": self.model_name,
        }
        if self.debug:
            self.main_tags["debug"] = "debug"
        tags = self.main_tags.copy()

        if self.extra_tags:
            for k, v in self.extra_tags.items():
                tags[k] = v

        self._description = description
        self._tags = tags

        if verbose:
            print(f"Description de '{self._name}' :")
            print(f"\t{self.main_description}")
            if self.extra_description:
                print(f"\t{self.extra_description}")
            print("Tous les Tags :")
            for k, v in self._tags.items():
                print(f"\t'{k}' : {v}")
        return self._description, self._tags

    def init(self, verbose=True, **kwargs):
        # On vérifie si les arguments sont valides (correspondent à des attributs publics de self)
        valid_args = [a for a in kwargs.keys() if a in self.valid_args]
        invalid_args = [arg for arg in kwargs.keys() if arg not in valid_args]

        if invalid_args:
            print(
                f"WARNING methode init de la classe {self.__class__.__name__} : les arguments {invalid_args} sont invalides et n'ont pas été initialisés"
            )
            print("\tArguments valides :", valid_args)

        # On met à jour tous les arguments valides (même s'il figurait des invalides)
        for a in valid_args:
            # self.__setattr__(a, kwargs[a])
            self.__dict__[a] = kwargs[a]
        self._init_name(verbose=verbose)
        self._init_tags(verbose=verbose)


class ExpMlFlow:
    def __init__(self, task, model_name, debug=False):
        self.input_dir = DATA_INTERIM
        self.train_filename = "00_v3_train.csv"
        # self.output_dir = os.path.join(MODEL_DIR, model_name)
        self.output_dir = MODEL_DIR

        self.meta = ExpMetaData(task, model_name, debug=debug)

        self.pipe_preprocess = None
        self.sampling = None
        self.uri_mlflow = f"http://{HOST_MLFLOW}:{PORT_MLFLOW}"
        # Attributs à construire obligatoirement avec initialisation
        self.X = None
        self.y = None
        self.random_state = VAL_SEED
        self._mlflow_id = None
        self._initialized = False
        # self.parent_run_id = None
        self.device = "cuda"

        # self.debug = debug
        self.debug_n_predictors = 20
        self.debug_frac = 0.1
        self._directories_ok = False
        # Service postgesql
        self._check_pg = False
        self._services_ok = False

    # attributs publics de l'expérience (sans ceux de l'objet meta), utilisé pour l'initialisation
    @property
    def valid_args_exp(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def check_postgresql_server(
        self,
        host="localhost",
        port=PORT_PG,
        dbname="optuna_db",
        user=USER_PG,
        password=PASSWORD_PG,
        verbose=True,
    ):
        try:
            connection = psycopg2.connect(
                host=host, port=port, dbname=dbname, user=user, password=password
            )
            connection.close()
            if verbose:
                print("Connexion Postgresql OK")
                return True
        except OperationalError as e:
            self._initialized = False
            print(
                "La connexion PostgreSQL a échoué. Vérifier que le service est démarré"
            )
            print(
                "Pour démarrer le service, vous pouvez utiliser la méthode .start_postresql_server()"
            )
            print("Error:", e)
            return
        """
        On peut vérifier le statut du serveice en ligne de commande :
        sudo service postgresql status (mdp depuis secret)
        :q pour sortir
        Redémarrer :
        sudo service postgresql restart
        """

    def check_mlflow_server(self, verbose=True):
        # Use the fluent API to set the tracking uri and the active experiment
        mlflow.set_tracking_uri(self.uri_mlflow)
        try:
            response = requests.get(self.uri_mlflow)
            if response.status_code == 200:
                print("Connexion MLFlow Ok")
                return True
        except requests.exceptions.RequestException as e:
            self._initialized = False
            print("La connexion MLFlow a échoué.")
            if verbose:
                print("Pour le démarrer, utilisez la méthode .start_mlflow_server()")
                print("Error:", e)
            return

    def check_services(self, verbose=True):
        if verbose:
            print("Vérification des services")
        mlflow_ok = False
        if self.check_mlflow_server(verbose=False):
            mlflow_ok = True
        if not mlflow_ok:
            if self.start_mlflow_server():
                mlflow_ok = True
        pg_ok = True
        if self._check_pg:
            pg_ok = False
            if self.check_postgresql_server(verbose=verbose):
                pg_ok = True
        self._services_ok = mlflow_ok * pg_ok
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
        print("Démarrage du serveur MLFlow...")
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Allow some time for the server to start
        time.sleep(5)
        if check_connect:
            return self.check_mlflow_server()
        return process

    def check_directories(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            print(f"Répertoire {self.output_dir} créé")
        self._directories_ok = True
        return

    # Appeler cette fonc avant même init_meta ? ou bloquer debug dans init_data ?
    # Lecture des données AVANT resampling de type smote etc.
    def read_data(
        self,
        verbose=True,
    ):

        ##################### Lecture de toutes les données de train (=avec target ET déjà partagé pour laisser un vrai test.csv)

        if verbose:
            print(
                f"Lecture des données à partir du fichier {self.train_filename}, device : {self.device}"
            )

        all_train = cudf.read_csv(os.path.join(self.input_dir, self.train_filename))
        if self.device != "cuda":
            all_train = all_train.to_pandas()
            # S'il y a des nan dans des bool d'un pandas.dataframe ça donnera des object, on les convertit en float
            object_features = all_train.select_dtypes("object").columns
            all_train[object_features] = all_train[object_features].astype("float64")

        # On place la target à part
        self.X = all_train.drop("TARGET", axis=1)
        self.y = all_train["TARGET"]

        # Si debug est activé, on réduit le nombre de prédicteurs et le nombre de lignes
        if self.meta.debug:
            print(
                f"Mode DEBUG activé. Nombre de prédicteurs : {self.debug_n_predictors}, Echantillonnage des lignes : {self.debug_frac:.0%}"
            )
            self.reduce_data(
                n_predictors=self.debug_n_predictors,
                frac_sample=self.debug_frac,
                verbose=False,
            )

        # On affiche les informations concernant X et y
        memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
        print(
            f"Forme de X : {self.X.shape}, type : {type(self.X)}, conso mémoire : {memory_consumed_mb:.0f} Mo"
            f"Forme de y : {self.y.shape}, type : {type(self.y)}"
        )

    # Réduit le nombre de lignes et de colonnes (utilisé pour le mode debug)
    def reduce_data(self, n_predictors=None, frac_sample=None, verbose=True):

        # Si n_predictors ou frac_sample sont précisés on en tient compte, sinon on prend les valeurs par défaut
        if frac_sample:
            self.debug_frac = frac_sample
        if n_predictors:
            self.debug_n_predictors = n_predictors

        # Si le nombre de prédicteurs à conserver est trop grand, on les prends tous
        all_predictors = [f for f in self.X if f not in ["SK_ID_CURR", "TARGET"]]
        if self.debug_n_predictors > len(all_predictors):
            self.debug_n_predictors = len(all_predictors)

        # On réduit le nombre de prédicteurs en gardant SK_ID_CURR
        X = self.X[["SK_ID_CURR"] + all_predictors[: self.debug_n_predictors]]
        self.X = X

        del all_predictors
        gc.collect()

        # On réduit le nombre de lignes en gardant seulement un échantillon stratifié
        X_sampled, _, y_sampled, _ = train_test_split_nan(
            self.X,
            self.y,
            test_size=1 - self.debug_frac,
            random_state=self.random_state,
            # C'est toujours stratifié par y
            # stratify=self.y,
        )

        self.X = X_sampled
        self.y = y_sampled
        if verbose:
            memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
            print(
                f"Forme de X : {self.X.shape}, type : {type(self.X)}, conso mémoire : {memory_consumed_mb:.0f} Mo"
            )

    def init_meta(self, **kwargs):
        self.meta.init(**kwargs)

    # Appeler init_data avant cette fonc
    def init_resampling(self):
        # [TODO] Implémenter le resampling ex smote
        return

    # Appeler init_config avant cette fonc
    def create_or_load_experiment(self, verbose=True):
        if not self._initialized:
            print("Iniatialisez d'abord la configuration avec .init_config():")
            return
        if not self._services_ok:
            print("La création de l'expérience nécessite que les services soient ok")
            return
        tags = self.meta._tags
        tags["mlflow.note.content"] = self.meta._description

        # On vérifie si l'expérience existe déjà
        existing_experiment = mlflow.get_experiment_by_name(self.meta._name)

        load_or_create = "Création de "
        if existing_experiment:
            experiment_id = existing_experiment.experiment_id
            load_or_create = "Chargement de "
        else:
            experiment_id = mlflow.create_experiment(
                self.meta._name,
                artifact_location=self.output_dir,
                tags=tags,
            )
        self._mlflow_id = experiment_id
        if verbose:
            print(
                f"{load_or_create}l'expérience MLFlow '{self.meta._name}', ID = {self._mlflow_id}"
            )
        # self.create_parent_run()
        return self._mlflow_id

    def init_config(self, verbose=True, **kwargs):
        # Si les services ne sont pas ok on les vérifie et s'ils ne sont toujours pas ok on sort:
        if not self._services_ok:
            self.check_services(verbose=verbose)
        if not self._services_ok:
            print("Pour initialiser la configuration, les servies doivent être ok")
            return

        # La liste des paramètres possibles à passer dans kwargs sont
        # les attributs qui ne sont pas protégés ni privés dans l'objet experience ou dans l'objet meta
        args_to_init_in_meta = [
            arg for arg in kwargs.keys() if arg in self.meta.valid_args
        ]
        args_to_init_in_exp = [
            arg for arg in kwargs.keys() if arg in self.valid_args_exp
        ]
        invalid_args = [
            arg
            for arg in kwargs.keys()
            if arg not in self.valid_args_exp and arg not in self.meta.valid_args
        ]
        if verbose:
            print(f"Initialisation")

        # On met à jour les paramètres autorisés pour meta et on initialise meta
        for a in args_to_init_in_meta:
            self.meta.__dict__[a] = kwargs[a]
        self.meta.init(verbose=verbose)

        # On met à jour les paramètres qui sont autorisés dans l'experience
        for a in args_to_init_in_exp:
            self.__dict__[a] = kwargs[a]

        if invalid_args:
            print(
                f"Warning : Les paramètres {invalid_args} ne sont pas valides. Liste des paramètres possibles :"
            )
            print(self.valid_args_exp + self.meta.valid_args)

        self.check_directories()

        if invalid_args:
            return False
        else:
            self._initialized = True
            return self._initialized


class ExpSearch(ExpMlFlow):
    def __init__(
        self,
        model_name="cuml",
        debug=False,
        metric="auc",
        direction="maximize",
    ):
        super().__init__(task="search", model_name=model_name, debug=debug)
        self.pipe_preprocess = None
        self.sampling = None
        self.metric = metric
        self.direction = direction
        self.meta.suffix1 = direction
        self.meta.suffix2 = metric
        self.meta.extra_tags = {"direction": self.direction, "metric": self.metric}

        # Par défaut dans optuna, on loggue les trials à l'écran
        # Pour ne loguer que les trials optuna.logging.WARNING
        # L'affichage des infos ou non sera véritablement effectué lors de l'optimisation en fonction du nombre de trials dans les classes enfantes
        self.optuna_verbosity = optuna.logging.INFO
        optuna.logging.set_verbosity(self.optuna_verbosity)
        self.sampler = optuna.samplers.TPESampler()
        self.pruner = optuna.pruners.HyperbandPruner(
            min_resource=10, max_resource=400, reduction_factor=3
        )
        self.storage = RDBStorage(
            f"postgresql://{USER_PG}:{PASSWORD_PG}@localhost:{PORT_PG}/optuna_db"
        )
        # Attributs à construire obligatoirement avec initialisation
        # self._study_name = ""
        self._mlflow_id = None
        self._study = None
        self._initialized = False
        self._parent_run_id = None
        # Turn off optuna log notes.
        # optuna.logging.set_verbosity(optuna.logging.WARN)
        # Les expéreince de recherche d'yperparamètres nécessitent postgresql en back-end
        self._check_pg = True
        # Par défaut on ne loggue pas dans mlflow le plot des params en parallel, cela dépend du modèle,
        # donc à définire dans classe enfant
        self._params_to_plot_in_parallel = []

    # Attributs publics de l'objet
    @property
    def valid_args_exp(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    # Appeler meta.init() avant cette fonc, et check_service (pg doit être démarré)
    def create_or_load_study(
        self,
    ):

        study = optuna.create_study(
            study_name=self.meta._name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            # On garde True pour load_if_exists au sujet de la study optuna (à ne pas confondre avec l'expérience mlflow)
            # Donc si l'expérience de recherche n'a pas été trackée sur mlflow mais faite uniquement dans optuna,
            # alors elle existe et on la continue. Cependant ce ne sera a priori jamais le cas.
            load_if_exists=True,
        )
        # [TODO] si le temps mettre un warning si la study existe déjà (voir si nécessaire)
        self._study = study
        return self._study

    # Prendre plutôt initi_config du parent
    def init(self, verbose=True, **kwargs):
        self.meta.init(**kwargs)
        # La liste des paramètres possibles à passer dans kwargs sont
        # les attributs qui ne sont pas protégés ni privés et qui ne sont pas des méthodes
        # Dans cet objet ou dans la l'objet meta imbriqué
        valid_args_meta = self._meta.valid_args

        # valid_args = [arg for arg in kwargs.keys() if arg in public_attributes]
        valid_args = [arg for arg in kwargs.keys() if arg in self.valid_args]
        invalid_args = [arg for arg in kwargs.keys() if arg not in valid_args]

        # On met à jour les paramètres qui sont autorisés
        for k in valid_args:
            v = kwargs[k]
            self.__setattr__(k, v)

        if verbose:
            # print(f"Les paramètres {valid_args} ont été mis à jour")
            print(f"Initialisation")
        if invalid_args:
            print(
                f"Les paramètres {invalid_args} ne sont pas valides. Liste des paramètres possibles :"
            )
            print(self.valid_args)

        # On ne veut pas ré-initialiser les data pour rien (la lecture évitée)
        # donc on récupère les arguments de kwargs qui sont aussi dans la signature de init_data
        # on n'exécute init_data que si les df X ou y sont vides ou si des arguments de data ont été passés dans kwargs, même s'ils ne sont pas différents d'avant
        data_signature = inspect.signature(self.init_data)
        data_args = [param.name for param in data_signature.parameters.values()]
        data_args_to_init = [a for a in kwargs.keys() if k in data_args]
        if self.X is None or self.y is None or data_args_to_init:
            self.init_data(verbose=False)
        print(
            f"Forme de X : {self.X.shape}, conso mémoire : {get_memory_consumed(self.X, verbose=False):.0f} Mo"
        )
        self.check_directories()
        self.check_services(verbose=True)
        # self.meta._init_name(verbose=False)
        # self.meta._init_tags(verbose=False)
        self.meta.init()
        self.init_data()
        optuna.logging.set_verbosity(self.optuna_verbosity)

        if invalid_args:
            return False
        else:
            self._initialized = True
            return self._initialized

    # Appeler les init avant les create
    # On surcharge car pour une experience de recherche il faut créer un objet optuna study
    def create_or_load(self):
        if self._initialized:
            id_experiment = self.create_or_load_experiment()
            study = self.create_or_load_study()
            return id_experiment, study
        else:
            print("Initialisez d'abord la config avec init.config()")
            return None

    def create_parent_run(self):
        # On crée un run parent qui va contenir tous les runs (un run enfant par trial)
        parent_run = mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name="Study",
            tags=self.tags,
            # description="Résultats de la recherche d'hyperparamètres",
        )
        self._parent_run_id = parent_run.info.run_id
        mlflow.end_run()

    def create_best_run(self):
        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=f"T_{self._study.best_trial.number}_best_of_{len(self._study.get_trials())}_trials",
        ):
            mlflow.log_params(self._study.best_trial.params)
            mlflow.log_metric(self.metric, self._study.best_trial.value)
            fig = optuna.visualization.plot_param_importances(self._study)
            fig.write_html(os.path.join(self.output_dir, "hyperparam_importance.html"))
            mlflow.log_artifact(
                os.path.join(self.output_dir, "hyperparam_importance.html")
            )
            # On ne logue dans ml flow le parallel_plot que si une liste est définie (change en fonction deu modèle)
            if self._params_to_plot_in_parallel:
                fig = optuna.visualization.plot_parallel_coordinate(
                    self._study,
                    params=self._params_to_plot_in_parallel,
                )
                fig.write_html(
                    os.path.join(self.output_dir, "parallel_coordinate.html")
                )
                mlflow.log_artifact(
                    os.path.join(self.output_dir, "parallel_coordinate.html")
                )

            # On loggue dans mlflow les visualisations qui ne dépendent pas du modèle
            fig = optuna.visualization.plot_optimization_history(self._study)
            fig.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            mlflow.log_artifact(
                os.path.join(self.output_dir, "optimization_history.html")
            )

    # Inutile, on va plutôt faire un run parent
    def track_best_run(self):
        # find the best run, log its metrics as the final metrics of this run.
        # client = MlflowClient()

        # Récupération de tous les runs de l'experiment
        runs = mlflow.search_runs(experiment_ids=[self._mlflow_id])
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
            [self._mlflow_id], f"tags.mlflow.parentRunName == 'best'"
        )"""

        # Filtrer les runs qui ont le run parent désiré
        # nested_runs = runs[runs["tags.mlflow.parentRunName"] == parent_run_name]
        # nested_runs = [runs[runs["tags.mlflow.parentRunId"]] == self._parent_run_id]

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

    """# Permet de suggérer des float en en proposant plus autour de 0.5 si
    # Si les params alpha et beta sont égaux et low=0 et high=1
    def suggest_beta(self, param_name, low=0, high=1, alpha=2.0, beta_param=2.0):
        # Sample a value from the beta distribution
        value = beta.rvs(alpha, beta_param)
        # Scale the value to the desired range [low, high]
        scaled_value = low + (high - low) * value
        return scaled_value
"""


def delete_experiments_with_prefix(prefix):
    client = MlflowClient()
    # experiments = client.list_experiments()
    experiments = mlflow.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )

    for experiment in experiments:
        if experiment.name.startswith(prefix):
            print(
                f"Deleting experiment: {experiment.name} (ID: {experiment.experiment_id})"
            )
            client.delete_experiment(experiment.experiment_id)


def del_studies_from_mlflow():

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
                print(f"No Optuna study found for MLflow experiment '{exp.name}'")
    print("Liste des studys dans mlflow :", tagged_experiments)"""

    # On recherche toute les expériences MLFlow dont le nom commence par search
    mlflow_tracking_uri = f"{HOST_MLFLOW}:{PORT_MLFLOW}"
    # client = MlflowClient()
    all_experiments = mlflow.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    search_experiments = [
        exp for exp in all_experiments if exp.name.startswith("search_")
    ]

    # On dresse la liste de toutes les études optuna
    storage = f"postgresql://{USER_PG}:{PASSWORD_PG}@localhost/optuna_db"
    all_studies = optuna.study.get_all_study_names(storage=storage)

    # La liste des études à supprimer dans optuna sont celles qui ne figurent pas dans MLFlow
    studies_to_delete = [
        study for study in all_studies if study not in search_experiments
    ]

    print(f"Supression de {len(studies_to_delete)} études optuna :")
    print(studies_to_delete)
    for study in studies_to_delete:
        optuna.delete_study(study_name=study, storage=storage)
    return
