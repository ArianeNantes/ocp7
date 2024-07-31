import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import os
import shutil
import gc
import time
import joblib
import subprocess
import inspect
from PIL import Image
from IPython.display import display
from plotly.io.kaleido import scope

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
import cudf
import cuml
import cupy as cp
from cuml.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from copy import deepcopy
import warnings

from .p7_evaluate import plot_recall_mean, cuml_cross_evaluate
from src.p7_util import timer, clean_ram, format_time, ask_confirmation
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

from src.p7_preprocess import (
    train_test_split_nan,
    check_variances,
    balance_smote,
    balance_nearmiss,
)
from src.p7_simple_kernel import get_memory_consumed
from src.p7_feature_selection import (
    cluster_features,
    plot_permutation_importance,
    plot_dendro,
    build_linkage_matrix,
    cluster_features_from_linkage_matrix,
    plot_top_correlations,
    build_corr_matrix,
    get_features_correlated_above,
)

from src.p7_evaluate import plot_evaluation_scores


# Classe pour fixer les méta données d'une expérience MLFlow :
# Le nom de l'expérience, les tags et la description
class ExpMetaData:
    def __init__(self, task, model_name, balance="none", num=0, debug=False):
        self.num = num
        self.task = task
        self.model_name = model_name
        self.suffix1 = ""
        self.suffix2 = ""
        self.balance = balance
        # Par défaut pas de resampling
        self.balance_k_neighbors = 0
        # Avec smote le rééquilibrage parfait (à 50% de défaut) nécessite trop de VRAM
        if balance == "smote":
            self.sampling_strategy = 0.7
        else:
            self.sampling_strategy = 0

        self.debug = debug
        self.continue_if_exist = False
        self._name = None
        self.main_description = "Description principale de l'expérience mlflow"
        self.description2 = ""
        self.description3 = ""
        self._description = ""
        self._tags = {}
        self.main_tags = {
            "task": self.task,
            "model": self.model_name,
            "debug": self.debug,
            "balance": self.balance,
        }
        self.extra_tags = {}

    @property
    def valid_args(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    def _init_neighbors(self):
        if self.balance_k_neighbors == 0:
            # Valeurs par défaut pour le resampling
            if self.balance == "smote":
                self.balance_k_neighbors = 5
            elif self.balance == "nearmiss":
                self.balance_k_neighbors = 3

        return

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
            print(f"\tNom de l'expérience{detail} :")
            print(f"\t\t{self._name}")
        return name, existing_experiment

    # Appeler cette fonc après init_name, pour changer les tags ou la description, appeler init
    def _init_tags(
        self,
        verbose=True,
    ):

        self.main_tags = {
            "task": self.task,
            "model": self.model_name,
            "balance": self.balance,
        }
        if self.debug:
            self.main_tags["debug"] = "debug"
        tags = self.main_tags.copy()

        if self.extra_tags:
            for k, v in self.extra_tags.items():
                tags[k] = v

        if self._description:
            tags["mlflow.note.content"] = self._description

        self._tags = tags

        return self._tags

    def _init_description(self, main_description=None, verbose=True):
        # La première ligne de description concerne la tâche
        # Correspondance task et et description
        dic_description = {
            "search": "Recherche d'hyperparamètres",
            "corr": "Suppression de features trop similaires 2 à 2",
            "permut": "Suppression de features par permutation importance",
            "eval": "Evaluation de modèle",
            "test": "Test de modèle",
        }
        if not main_description:
            self.main_description = dic_description[self.task]
        else:
            self.main_description = main_description
        self._description = f"{self.main_description}"

        # La 2ème ligne concerne des détails éventuels
        if self.description2:
            self._description = f"{self._description}\n{self.description2}"

        # La 3ème ligne concerne les réquilibrage du data_set
        self.description3 = f"Rééquilibrage dataset : {self.balance}"
        # On n'écrit pas le nombre de voisins car on le recherche
        """if self.balance != "none":
            self.description3 = (
                self.description3 + f" {self.balance_k_neighbors} voisins"
            )"""
        if self.description3:
            self._description = f"{self._description}\n{self.description3}"
        """if verbose:
            print(f"\tDescription de '{self._name}' :")
            print(f"\t\t{self.main_description}")
            if self.description2:
                print(f"\t\t{self.description2}")
            if self.description3:
                print(f"\t\t{self.description3}")"""
        return self._description

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
        # self._init_name(verbose=verbose)
        self._init_neighbors()
        self._init_description(verbose=verbose)
        self._init_tags(verbose=verbose)
        self._init_name(verbose=verbose)


class ExpMlFlow:
    def __init__(self, task, model_name, debug=False):
        self.input_dir = DATA_INTERIM
        self.train_filename = "00_v3_train.csv"
        # self.output_dir = os.path.join(MODEL_DIR, model_name)
        self.output_dir = "mlflow_artifacts"

        self.meta = ExpMetaData(task, model_name, debug=debug)

        self.pipe_preprocess = None
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
        # Traçage mlflow du dataset
        self._mlflow_dataset = None
        self._run_id = None

    # attributs publics de l'expérience (sans ceux de l'objet meta), utilisé pour l'initialisation
    @property
    def valid_args_exp(self):
        return [k for k in self.__dict__.keys() if not k.startswith("_")]

    @property
    def _resampling_str(self):
        resampling = f"Ratio défaut {self.get_ratio_default():.1%}"
        if self.meta.balance != "none":
            resampling = resampling + f" - rééquilibrage {self.meta.balance.upper()}"
        return resampling

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
                print("\tConnexion Postgresql OK")
                return True
        except OperationalError as e:
            self._initialized = False
            print(
                "\tLa connexion PostgreSQL a échoué. Vérifier que le service est démarré"
            )
            print(
                "\tPour démarrer le service, vous pouvez utiliser la méthode .start_postresql_server()"
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
                print("\tConnexion MLFlow Ok")
                return True
        except requests.exceptions.RequestException as e:
            self._initialized = False
            print("\tLa connexion MLFlow a échoué.")
            if verbose:
                print("\tPour le démarrer, utilisez la méthode .start_mlflow_server()")
                print("Error:", e)
            return

    def check_services(self, verbose=True):
        if verbose:
            print("Vérification des services...")
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
                f"Lecture des données...\n\tFichier : {self.train_filename},\n\tdevice : {self.device.upper()}"
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
                f"\tMode DEBUG activé. Nb prédicteurs : {self.debug_n_predictors}, Echantillonnage lignes : {self.debug_frac:.0%}"
            )
            self.reduce_data(
                n_predictors=self.debug_n_predictors,
                frac_sample=self.debug_frac,
                verbose=False,
            )

        self.clean_read_data(drop_null_var=True, verbose=True)

        # On affiche les informations concernant X et y
        memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
        print(
            f"\tForme de X : {self.X.shape}, type : {type(self.X)}, conso mémoire : {memory_consumed_mb:.0f} Mo\n"
            f"\tForme de y : {self.y.shape}, type : {type(self.y)}\n"
            f"\tRatio initial de défauts : {self.get_ratio_default():.1%}"
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

    def clean_read_data(self, drop_null_var=True, verbose=True):
        # Nettoie les données lues
        if verbose:
            print("Nettoyage des données...")
        # S'il y a des features indésirables (par exemple 'Unnamed: 0' issue d'un oubli d'ignorer l'index en sauvegardant etc.) on les supprime
        irrelevant_features = sel_var(
            self.X.columns, include_pattern=r"\bUnnamed", verbose=False
        )
        irrelevant_features = irrelevant_features + sel_var(
            self.X.columns, include_pattern=r"\bindex", verbose=False
        )
        irrelevant_features = irrelevant_features + [
            f for f in self.X.columns if f in ["SK_ID_BUREAU", "SK_ID_PREV"]
        ]

        # Les features de variance nulle peuvent être considérées comme indésirables
        if drop_null_var:
            features_null_var = check_variances(
                self.X, raise_error=False, verbose=False
            )
            irrelevant_features = irrelevant_features + features_null_var

        if irrelevant_features:
            self.X = self.X.drop(irrelevant_features, axis=1)
            if verbose:
                print(
                    f"\t{len(irrelevant_features)} features supprimées :",
                    irrelevant_features,
                )
                # print(f"\t{irrelevant_features}")

            # print("Nouvelle forme de X", self.X.shape)

    def get_pipe_str(self):
        if self.pipe_preprocess:
            step_classes = [
                step[1].__class__.__name__ for step in self.pipe_preprocess.steps
            ]
        else:
            step_classes = []
        return f"{step_classes}"

    def print_preprocess(self, n_tabs=1):
        # print(f"[debug] fonc print_preprocess n_tabs {n_tabs}")
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        """print(f"{tab_level0}Preprocess :")
        tab_level_1 = tab_level0 + '\t'"""

        # print(f"{tab_level_1}Source : {self.train_filename}")

        if self.pipe_preprocess:
            step_classes = [
                step[1].__class__.__name__ for step in self.pipe_preprocess.steps
            ]
            print(f"{tab_level0}pipe_preproccess : {step_classes}")
        if self.meta.balance != "none":
            print(
                f"{tab_level0}Rééquilibrage dataset {self.meta.balance.upper()} {self.meta.balance_k_neighbors} voisins"
            )
        if self.pipe_preprocess is None and self.meta.balance == "none":
            print(f"{tab_level0}Aucun prétraitement")

    def init_meta(self, **kwargs):
        self.meta.init(**kwargs)

    """# Appeler init_data avant cette fonc
    def init_resampling(self):
        # [TODO] Implémenter le resampling ex smote
        return"""

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
            artifact_location = os.path.join(self.output_dir, self.meta._name)
            os.makedirs(artifact_location, exist_ok=True)
            experiment_id = mlflow.create_experiment(
                self.meta._name,
                artifact_location=artifact_location,
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
            print("Pour initialiser la configuration, les services doivent être ok")
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

        # On met à jour les paramètres autorisés pour meta et on initialise meta
        for a in args_to_init_in_meta:
            self.meta.__dict__[a] = kwargs[a]
            # print("in meta", a, self.meta.__dict__[a])
            dic_arg_meta = {
                k: v for k, v in kwargs.items() if k in args_to_init_in_meta
            }
        self.meta.init(**dic_arg_meta, verbose=False)

        # On met à jour les paramètres qui sont autorisés dans l'experience
        for a in args_to_init_in_exp:
            self.__dict__[a] = kwargs[a]

        # self.meta.init(verbose=True)
        # self.meta.init(**dic_arg_meta, verbose=False)

        self.check_directories()

        # On imprime la config
        if verbose:
            print(f"Configuration...")
            self.print_name(n_tabs=1)
            self.print_tags(n_tabs=1)
            self.print_description(n_tabs=1)
            self.print_dataset(n_tabs=1)
            self.print_preprocess(n_tabs=2)
            self.print_debug(n_tabs=2)

        if invalid_args:
            print(
                f"Warning : Les paramètres {invalid_args} ne sont pas valides. Liste des paramètres possibles :"
            )
            print(self.valid_args_exp + self.meta.valid_args)
            self._initialized = False
        else:
            self._initialized = True

        """if invalid_args:
            return False
        else:
            self._initialized = True
            return self._initialized"""

    def create_mlflow_dataset(self):
        warnings.simplefilter(action="ignore", category=UserWarning)
        # source = os.getcwd(), os.path.join(self.input_dir, self.train_filename)
        if isinstance(self.X, pd.DataFrame):
            X = self.X
        else:
            X = self.X.to_pandas()
        if isinstance(self.y, pd.Series):
            y = self.y
        else:
            y = self.y.to_pandas()
            y.name = "TARGET"
        dataset = pd.concat([X, y], axis=1)

        self._mlflow_dataset = mlflow.data.from_pandas(
            dataset,
            name=self.train_filename,
        )

    def track_dataset(self, context=""):
        if not self._mlflow_dataset:
            print("Créez d'abord le dataset mlflow avev .create_mlflow_dataset()")
            return
        if not context:
            if self.meta.balance != "none":
                context = self.meta.balance.upper()
            else:
                context = "unbalanced"
        mlflow.log_input(self._mlflow_dataset, context=context)

    def save_data(
        self, version=None, filename=None, suffix="_train_permuted", replace=False
    ):
        if self.X is None or self.y is None:
            print("Aucune donnée à sauvegarder")
            return
        dataset = self.X.copy()
        dataset["TARGET"] = self.y.to_numpy()

        if not filename:
            if not version:
                input_splits = self.train_filename.split("_")
                input_version = input_splits[1]
                imput_version_num = int(input_version[1:])
                version = imput_version_num + 1
            output_version = f"v{version}"
            output_train_filename = f"{input_splits[0]}_{output_version}{suffix}.csv"
        else:
            output_train_filename = filename
        pathfile = os.path.join(self.input_dir, output_train_filename)
        save = True
        if not replace:
            if os.path.exists(pathfile):
                print(f"'{pathfile}' existe déjà. Modifiez le nom du fichier")
                save = False
            else:
                save = True
        if save:
            dataset.to_csv(pathfile, index=False)
            print(f"Données sauvegardées dans {pathfile}")

    # Sauvegarde l'expérience comme un objet pkl
    # Ecrase un fichier existant sans rien demander
    def save_experiment(self, root_dir=MODEL_DIR, subdir="", verbose=True):
        if subdir:
            path = os.path.join(root_dir, subdir)
        else:
            path = os.path.join(root_dir, self.meta.model_name)

        # Si le répertoire n'existe pas, on le crèe
        if not os.path.exists(path):
            os.mkdir(path)
        filename = os.path.join(path, f"{self.meta._name}.pkl")
        joblib.dump(self, filename)
        if verbose:
            print(f"Expérience sauvegardée dans {filename}")

    def get_ratio_default(self):
        if self.y is not None:
            ratio_default = self.y.value_counts(normalize=True)[1]
            return ratio_default
        else:
            print("Les données n'ont pas été itialisées")
            return

    # Retourne l'ID d'un run à partir de son nom
    def get_run_id_from_name(self, run_name):
        return get_run_id_from_name(experiment_id=self._mlflow_id, run_name=run_name)

    def download_run_artifacts(self, run_name):
        run_id = self.get_run_id_from_name(run_name)
        if not run_id:
            print(
                f"Run {run_name} non trouvé dans l'expérience {self.meta._name} (ID {self._mlflow_id})"
            )
        else:
            # On liste tous les artifacts du run
            client = mlflow.tracking.MlflowClient()
            artifact_list = client.list_artifacts(run_id)
            artifact_paths = [artifact.path for artifact in artifact_list]
            print(f"Artifacts du run {run_name} :")
            print(artifact_paths)
            # Télécharger les artefacts dans un répertoire local temporaire
            local_dir = os.path.join(self.output_dir, "download")
            os.makedirs(local_dir, exist_ok=True)

            for artifact_path in artifact_paths:
                local_path = client.download_artifacts(run_id, artifact_path, local_dir)
                print(f"Artifact {artifact_path} downloaded to {local_path}")

                # Afficher les images
                if artifact_path.endswith((".png", ".jpg", ".jpeg")):
                    img = Image.open(local_path)
                    display(img)

                elif artifact_path.endswith(".json"):
                    # Lire et afficher le contenu des autres types de fichiers si nécessaire
                    with open(local_path, "r") as file:
                        content = file.read()
                        print(f"Content of {artifact_path}:\n{content}\n")

    # Affiche les artifacts d'un run sans les télécharger. Lit directement sur le disque
    def get_artifacts(self, run_id=None):
        if not run_id:
            run_id_to_show = self._run_id
        else:
            run_id_to_show = run_id

        if not run_id_to_show:
            print("Aucun run indiqué comme contenant des artifacts")
            return

        # On liste tous les artifacts du run
        client = mlflow.tracking.MlflowClient()
        artifact_list = client.list_artifacts(run_id_to_show)
        artifact_paths = [artifact.path for artifact in artifact_list]

        runs = mlflow.search_runs(experiment_ids=self._mlflow_id)
        artifact_uri = runs.loc[
            runs["run_id"] == run_id_to_show, "artifact_uri"
        ].values[0]
        run_name = runs.loc[
            runs["run_id"] == run_id_to_show, "tags.mlflow.runName"
        ].values[0]
        print(f"Artifacts du run {run_name} :")
        print("Path :", artifact_uri)
        print("Liste des artifacts :", artifact_paths)

        for artifact_path in artifact_paths:

            # Afficher les images
            if artifact_path.endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(os.path.join(artifact_uri, artifact_path))
                display(img)

            elif artifact_path.endswith(".json"):
                # Lire et afficher le contenu des autres types de fichiers si nécessaire
                with open(os.path.join(artifact_uri, artifact_path), "r") as file:
                    content = file.read()
                    print(f"Content of {artifact_path}:\n{content}\n")

    # Affiche les paramètres d'un run mlflow
    def get_params(self, run_id=None):
        if not run_id:
            run_id_to_show = self._run_id
        else:
            run_id_to_show = run_id

        if not run_id_to_show:
            print("Aucun run indiqué comme contenant des paramètres")
            return

        # On liste tous les runs de l'expérience, et on cherche celui correspondant à l'ID
        runs = mlflow.search_runs(experiment_ids=self._mlflow_id)
        param_columns = sel_var(runs.columns, include_pattern=r"\bparam", verbose=False)
        params = runs.loc[runs["run_id"] == run_id_to_show, param_columns]
        run_name = runs.loc[
            runs["run_id"] == run_id_to_show, "tags.mlflow.runName"
        ].values[0]
        # On renomme les colonnes en supprimant le préfixe 'params.'
        params.columns = params.columns.str.replace("params.", "", regex=False)
        print(f"Paramètres du run {run_name} :")
        display(params)
        return params

    # Affiche les métriques d'un run mlflow
    def get_metrics(self, run_id=None):
        if not run_id:
            run_id_to_show = self._run_id
        else:
            run_id_to_show = run_id

        if not run_id_to_show:
            print("Aucun run indiqué comme contenant des métriques")
            return

        # On liste tous les runs de l'expérience, et on cherche celui correspondant à l'ID
        runs = mlflow.search_runs(experiment_ids=self._mlflow_id)
        metric_columns = sel_var(
            runs.columns, include_pattern=r"\bmetric", verbose=False
        )
        metrics = runs.loc[runs["run_id"] == run_id_to_show, metric_columns]
        run_name = runs.loc[
            runs["run_id"] == run_id_to_show, "tags.mlflow.runName"
        ].values[0]
        # On renomme les colonnes en supprimant le préfixe 'metrics.'
        metrics.columns = metrics.columns.str.replace("metrics.", "", regex=False)
        print(f"Métriques du run {run_name} :")
        display(metrics)
        return metrics

    def balance(self, X, y, random_state=VAL_SEED, verbose=True):
        # Eventuel rééquilibrage avec SMOTE ou NearMiss
        _balance = self.meta.balance
        if _balance == "none":
            return X, y, []
        elif _balance == "smote":
            if self.meta.sampling_strategy == 0:
                self.meta.sampling_strategy = 0.7
            X_balanced, y_balanced = balance_smote(
                X,
                y,
                k_neighbors=self.meta.balance_k_neighbors,
                sampling_strategy=self.meta.sampling_strategy,
                random_state=random_state,
                verbose=verbose,
            )
            return X_balanced, y_balanced, []
        elif _balance == "nearmiss":
            X_balanced, y_balanced, null_var = balance_nearmiss(
                X,
                y,
                k_neighbors=self.meta.balance_k_neighbors,
                drop_null_var=False,
                verbose=verbose,
            )
            return X_balanced, y_balanced, null_var
        else:
            print(
                f"{_balance} est une valeur incorrecte pour balance. Les valeurs possibles sont 'none', 'smote' ou 'nearmiss'"
            )
            return

    def get_run_count(self):
        """Compte le nombre de runs existant dans l'expérience mlflow

        Returns:
            int: Nombre de runs existant dans l'expérience
        """
        if self._mlflow_id:
            runs = mlflow.search_runs(experiment_ids=[self._mlflow_id])
            return len(runs)
        else:
            print(
                "Créez d'abord l'expérience pour compter le nombre de runs à l'intérieur de celle-ci"
            )
            return

    def print_description(self, n_tabs=1):
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        # print(f"{tab_level0}Description Mlflow '{self.meta._name}' :")
        print(f"{tab_level0}Description Mlflow :")
        tab_str_level_1 = tab_level0 + "\t"
        print(f"{tab_str_level_1}{self.meta.main_description}")
        if self.meta.description2:
            print(f"{tab_str_level_1}{self.meta.description2}")
        if self.meta.description3:
            print(f"{tab_str_level_1}{self.meta.description3}")

    def print_tags(self, n_tabs=1):
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        print(f"{tab_level0}Tags Mlflow :")
        tab_str_level_1 = tab_level0 + "\t"
        for k, v in self.meta._tags.items():
            # On imprime tous les tags sauf la description mlflow
            if k != "mlflow.note.content":
                print(f"{tab_str_level_1}{k} : {v}")

    def print_dataset(self, n_tabs=1):
        # print(f"Fonc print_dataset n_tabs {n_tabs}")
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        print(f"{tab_level0}Données :")
        tab_str_level_1 = tab_level0 + "\t"

        print(f"{tab_str_level_1}Source : {self.train_filename}")

    def print_name(self, n_tabs=1):
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        print(f"{tab_level0}Nom de l'expérience :", self.meta._name)

    def print_debug(self, n_tabs=1):
        tab_level0 = ""
        for i in range(n_tabs):
            tab_level0 = tab_level0 + "\t"
        if self.meta.debug:
            print(f"{tab_level0}Mode DEBUG activé")


class ExpSearch(ExpMlFlow):
    def __init__(
        self,
        model_name="cuml",
        debug=False,
        metric="auc",
        direction="maximize",
        loss="binary",
    ):
        super().__init__(task="search", model_name=model_name, debug=debug)
        self.pipe_preprocess = None
        self.meta.balance = None
        self.metric = metric
        self.direction = direction
        self.loss = loss
        self.meta.suffix1 = direction
        self.meta.suffix2 = metric
        self.meta.extra_tags = {
            "direction": self.direction,
            "metric": self.metric,
            "loss": self.loss,
        }
        self.X = None
        self.y = None
        self._best_run_name = None

        # Par défaut dans optuna, on loggue les trials à l'écran
        # Pour ne loguer que les trials optuna.logging.WARNING
        # L'affichage des infos ou non sera véritablement effectué lors de l'optimisation en fonction du nombre de trials dans les classes enfantes
        self.optuna_verbosity = optuna.logging.INFO
        optuna.logging.set_verbosity(self.optuna_verbosity)
        self.sampler = optuna.samplers.TPESampler(VAL_SEED)
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
        self.n_folds = 4

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
            # pruner=self.pruner,
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
    """def init(self, verbose=True, **kwargs):
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
        self.meta.init()
        self.init_data()
        optuna.logging.set_verbosity(self.optuna_verbosity)

        if invalid_args:
            return False
        else:
            self._initialized = True
            return self._initialized"""

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

    def print_best_trial(self, print_params=True):
        # best_scores = self._study.best_trial.user_attrs.get("mean_scores")
        print("N° du Meilleur trial :", self._study.best_trial.number)
        print(f"Meilleur {self.metric} : {self._study.best_trial.value:.4f}")
        if print_params:
            self.print_suggested_params()

    def create_best_run(self, verbose=True):
        print("Création du meilleur run...")
        self._best_run_name = f"T_{self._study.best_trial.number}_best_of_{len(self._study.get_trials())}_trials"
        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=self._best_run_name,
        ) as best_run:
            self._run_id = best_run.info.run_id
            # mlflow.log_params(self._study.best_trial.params)
            mlflow.log_params(self.get_all_params())

            # Pour loguer le dataset
            self.create_mlflow_dataset()
            self.track_dataset()

            # Par défaut dans optuna, seule la métrique principale est enregistrée (on change ce comprtement dans les classe enfants, méthode objective)
            # Si tous les scores ont été enregistrés dans le trial, on les loggue tous dans mlflow, sinon on loggue juste la métrique à optimiser
            """if self._study.best_trial.user_attrs.get("mean_scores"):
                mlflow.log_metrics(self._study.best_trial.user_attrs.get("mean_scores"))"""
            try:
                mean_scores = self._study.best_trial.user_attrs.get("mean_scores")
                mlflow.log_metrics(mean_scores)
            except:
                mean_scores = None
                # "Pas de dico mean_scores pour ce best trial" (if study.user_attrs)
                mlflow.log_metric(self.metric, self._study.best_trial.value)

            if len(self._study.best_trial.params) > 1:
                fig = optuna.visualization.plot_param_importances(self._study)
                # fig.write_html(os.path.join(self.output_dir, "hyperparam_importance.html"))
                fig.write_image(
                    os.path.join(self.output_dir, "hyperparam_importance.png"),
                    format="png",
                )
                if verbose:
                    fig.show()
                mlflow.log_artifact(
                    os.path.join(self.output_dir, "hyperparam_importance.png")
                )
                # On ne logue dans ml flow le parallel_plot que si une liste est définie (change en fonction deu modèle)
                if self._params_to_plot_in_parallel:
                    fig = optuna.visualization.plot_parallel_coordinate(
                        self._study,
                        params=self._params_to_plot_in_parallel,
                    )
                    if verbose:
                        fig.show()
                    """fig.write_html(
                        os.path.join(self.output_dir, "parallel_coordinate.html")
                    )"""
                    width = scope.default_width * 2
                    fig.write_image(
                        os.path.join(self.output_dir, "parallel_coordinate.png"),
                        format="png",
                        # scale=2,
                        width=width,
                    )
                    mlflow.log_artifact(
                        os.path.join(self.output_dir, "parallel_coordinate.png")
                    )

            # On loggue dans mlflow les visualisations qui ne dépendent pas du modèle
            fig = optuna.visualization.plot_optimization_history(self._study)
            if verbose:
                fig.show()
            # fig.write_html(os.path.join(self.output_dir, "optimization_history.html"))
            fig.write_image(
                os.path.join(self.output_dir, "optimization_history.png"), format="png"
            )
            mlflow.log_artifact(
                os.path.join(self.output_dir, "optimization_history.png")
            )
            # On loggue dans mlflow la matrice Recall moyenne si mean_scores a été enregistré
            # name_best_run_in_plot = f"trial n°{self._study.best_trial.number} best of {len(self._study.get_trials())} trials"
            name_best_run_in_plot = f"Meilleur {self.metric} (trial n°{self._study.best_trial.number} / {len(self._study.get_trials())}) = {mean_scores[self.metric]:.2f}"
            """resampling = f"Ratio défaut {self.get_ratio_default():.1%}"
            if self.meta.balance != "none":
                resampling = (
                    resampling
                    + f" - rééquilibré à 50% avec {self.meta.balance.upper()}"
                )"""
            if self.n_folds == 1:
                title_recall = "Recall (% par ligne)"
            else:
                title_recall = f"Recall moyen sur {self.n_folds} folds (% par ligne)"
            if mean_scores:
                fig = plot_recall_mean(
                    tn=mean_scores["tn"],
                    fp=mean_scores["fp"],
                    fn=mean_scores["fn"],
                    tp=mean_scores["tp"],
                    title=title_recall,
                    subtitle=f"Optuna : '{self.meta._name}'\n{self._resampling_str}\n{name_best_run_in_plot}",
                    verbose=verbose,
                )
                # On sauvegarde la figure (la précédente sera écrasée, c'est volontarire)
                filepath = os.path.join(self.output_dir, "recall_mean_plot.png")
                fig.savefig(filepath, bbox_inches="tight")
                # On loggue dans mlflow
                mlflow.log_artifact(filepath)

            # On logue à l'écran les résultats pour la meilleure combinaison
            if verbose:
                print(
                    f"Meilleur {self.metric} (trial n°{self._study.best_trial.number} / {len(self._study.get_trials())}) = {mean_scores[self.metric]:.4f}"
                )

    def get_suggested_params(self):
        return self._study.best_trial.params

    def get_all_params(self):
        dic_params = self.get_suggested_params()
        dic_params["balance"] = self.meta.balance
        dic_params["sampling_strategy"] = self.meta.sampling_strategy
        dic_params["search_experiment"] = self.meta._name
        dic_params["loss"] = self.loss
        return dic_params

    def get_params_of_model(self):
        return {
            k: v
            for k, v in self.get_suggested_params().items()
            if k not in ["threshold_prob", "k_neighbors", "balance"] and not pd.isna(v)
        }

    def print_all_params(self):
        params = self.get_all_params()
        print("Tous les paraamètres :")
        for k in params.keys():
            print(f"\t{k} : {params[k]}")

    def print_params_of_model(self):
        params_of_model = self.get_params_of_model()
        print("Paramètres du modèle :")
        for k in params_of_model.keys():
            print(f"\t{k} : {params_of_model[k]}")

    def print_suggested_params(self):
        params = self.get_suggested_params()
        print("Paramètres suggérés :")
        for k in params.keys():
            print(f"\t{k} : {params[k]}")

    def counts_pruned_trials(self):
        # On récupère tous les trials de l'étude (deepcopy=False est + efficace en mémore car référence directe aux trials)
        trials = self._study.get_trials(deepcopy=False)
        n_pruned_trials = sum(
            1 for trial in trials if trial.state == optuna.trial.TrialState.PRUNED
        )
        return n_pruned_trials

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

    """def get_best_artifacts(self):
        self.get_run_artifacts(self._best_run_id)"""

    """# Permet de suggérer des float en en proposant plus autour de 0.5 si
    # Si les params alpha et beta sont égaux et low=0 et high=1
    def suggest_beta(self, param_name, low=0, high=1, alpha=2.0, beta_param=2.0):
        # Sample a value from the beta distribution
        value = beta.rvs(alpha, beta_param)
        # Scale the value to the desired range [low, high]
        scaled_value = low + (high - low) * value
        return scaled_value
"""


class ExpPermutation(ExpMlFlow):
    def __init__(
        self,
        model=LogisticRegression(),
        model_name="logreg",
        # threshold_prob=0.5,
        debug=False,
    ):
        super().__init__(task="permut", model_name=model_name, debug=debug)
        self.model = model
        self.pipe_preprocess = None
        # [TODO] corriger balance
        # self.sampling = None
        self.meta.balance = "none"
        self.metric = "auc"
        self.X = None
        self.y = None
        self._useless_features = None
        self._usefull_features = None
        self._run_id = None
        # self.threshold_prob = threshold_prob

    def run(self, n_folds=5, random_state=VAL_SEED):

        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=f"Permutation",
        ) as run:
            self._run_id = run.info.run_id
            mlflow.log_params(self.model.get_params())

            # Pour loguer le dataset
            self.create_mlflow_dataset()
            self.track_dataset()

            predictors = [
                f for f in self.X.columns if f not in ["TARGET", "SK_ID_CURR"]
            ]

            X_tmp = self.X[predictors[:2]].to_numpy()
            y_tmp = self.y.to_pandas()

            folds = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state,
            )

            # Le dictionnaire contenant les résultats contient une clef pour chaque feature.
            # Les valeurs sont les listes des importances (une importance par fold)
            dic_importances = {col: [] for col in predictors}
            baseline_scores = []

            for train_idx_array, valid_idx_array in folds.split(X_tmp, y_tmp):
                train_idx = train_idx_array.tolist()
                valid_idx = valid_idx_array.tolist()

                X_train = self.X.iloc[train_idx]
                X_val = self.X.iloc[valid_idx]
                if "SK_ID_CURR" in X_train.columns:
                    X_train = X_train.drop("SK_ID_CURR", axis=1)
                    X_val = X_val.drop("SK_ID_CURR", axis=1)

                y_train = self.y.iloc[train_idx]
                y_val = self.y.iloc[valid_idx]

                # print(f"\tPrétraitement")
                pipe = deepcopy(self.pipe_preprocess)
                X_train_processed = pipe.fit_transform(X_train)
                X_val_processed = pipe.transform(X_val)

                # Afin de repartir d'un modèle non fitté, on crée une nouvelle copie du modèle passé en paramètre
                clf = deepcopy(self.model)
                clf.fit(X_train_processed, y_train)

                # Probabilité d'appartenir à la classe default pour le jeu de validation
                y_score_val = clf.predict_proba(X_val_processed)[1]

                # Mesure sur le jeu de validation avec toutes les colonnes intactes
                baseline_score = cuml.metrics.roc_auc_score(y_val, y_score_val)
                baseline_scores.append(baseline_score)

                # Pour chaque colonne on mélange la colonne, et on calcule la différence de score avec la baseline
                processed_predictors = [
                    col for col in X_val_processed.columns if col not in ["SK_ID_CURR"]
                ]
                for col in processed_predictors:
                    X_permuted = X_val_processed.copy()

                    # Le fait de faire le sample sur 100% des données en ignorant l'index revient à effectuer un shuffle
                    X_permuted[col] = (
                        X_permuted[col]
                        .sample(
                            frac=1,
                            replace=False,
                            ignore_index=True,
                            random_state=random_state,
                        )
                        .to_numpy()
                    )

                    permuted_score = cuml.metrics.roc_auc_score(
                        y_val, clf.predict_proba(X_permuted)[1]
                    )
                    dic_importances[col].append(baseline_score - permuted_score)
                    # print("permuted_score", permuted_score)

                # Lors du pripe de preprocess, des colonnes ont pu être supprimées à cause d'une variance nulle
                deleted_columns_in_process = [
                    col for col in predictors if col not in processed_predictors
                ]
                for col in deleted_columns_in_process:
                    dic_importances[col].append(0)

            # on transforme en df, les features sont en colonnes et les folds en lignes
            importances = pd.DataFrame(dic_importances)

            # On loggue la liste des features de moindre importance, c'est à dire celles qui ont en moyenne un score inférieur ou égal à zéro
            importances_mean = importances.mean()
            useless_features = importances_mean[importances_mean <= 0].index.tolist()
            self._useless_features = useless_features
            dic_useless_features = {"useless_features": useless_features}
            mlflow.log_dict(dic_useless_features, "useless_features.json")
            # On logue les features utiles
            usefull_features = importances_mean[importances_mean > 0].index.tolist()
            self._usefull_features = usefull_features
            dic_usefull_features = {"dic_usefull_features": usefull_features}
            mlflow.log_dict(dic_usefull_features, "usefull_features.json")

            # On logue la moyenne des AUC scores avant permutation
            baseline_scores_mean = np.mean(baseline_scores)
            mlflow.log_metric("baseline AUC mean", baseline_scores_mean)

            # On logue le plot
            fig = plot_permutation_importance(
                importances,
                f"Permutation importance sur {n_folds} folds\n{self.meta._name}\n",
            )
            if fig:
                # On sauvegarde la figure (la précédente sera écrasée, c'est volontarire)
                filepath = os.path.join(self.output_dir, "permutation.png")
                fig.savefig(filepath)
                # On loggue dans mlflow
                mlflow.log_artifact(filepath)

            del pipe
            del clf
            gc.collect()
            print(f"{len(useless_features)} features sont inutiles :")
            print(useless_features)
        return useless_features

    def drop_useless_features(self, verbose=True):
        if self._useless_features:
            to_drop = [
                f for f in self._useless_features if f in self.X.columns.tolist()
            ]
            self.X = self.X.drop(to_drop, axis=1)
            if verbose:
                print(f"{len(to_drop)} features supprimées")
                memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
                print(
                    f"Forme de X : {self.X.shape}, type : {type(self.X)}, conso mémoire : {memory_consumed_mb:.0f} Mo"
                )


class ExpCorr(ExpMlFlow):
    def __init__(
        self,
        # cluster_corr pour similarité correlation, cluster_agglo pour agglomerative clustering direct
        # model_name="cluster_corr",
        debug=False,
    ):
        super().__init__(task="corr", model_name="cluster", debug=debug)
        self.train_filename = "00_v1_train.csv"
        self.pipe_preprocess = None
        self.meta.balance = None
        self.corr_coef = "pearson"
        self.threshold_corr = 0.98
        self.threshold_dist = 0.1
        self.agglo_method = "complete"
        self.X = None
        self.y = None
        # self.model_name = "corr"
        # On regroupe sur une métrique basée sur la corrélation
        self.metric = "1-|corr|"

        self.meta.suffix1 = self.corr_coef
        self.meta.suffix2 = self.metric
        self.meta.main_description = "Suppression des features trop corrélées 2 à 2"
        self.meta.description2 = f"Méthode clustering : {self.agglo_method} - métrique : {self.metric} {self.corr_coef.upper()}"
        self._features_correlated_above_threshold = None
        self._features_to_drop = None
        self._run_id = []

    def run(self, verbose=True):
        self._run_id = []
        X_balanced = self.preprocess(verbose=verbose)
        corr_matrix = build_corr_matrix(
            X_balanced, corr_coef=self.corr_coef, verbose=verbose
        )
        self.run_top_10_correlations(X_balanced, corr_matrix, verbose=verbose)
        self.run_features_corr_above(X_balanced, corr_matrix, verbose=verbose)
        self.run_redondant_features(X_balanced, verbose=verbose)

    def preprocess(self, verbose=True):
        predictors = [
            f for f in self.X.columns if f not in ["TARGET", "SK_ID_CURR", "Unnamed: 0"]
        ]
        X = self.X[predictors].copy()
        y = self.y.copy()

        # print(f"\tPrétraitement")
        if self.pipe_preprocess:
            X_processed = self.pipe_preprocess.fit_transform(X)
        else:
            X_processed = self.X

        # Eventuel rééquilibrage avec SMOTE ou NearMiss
        X_balanced, y_balanced, null_var = self.balance(X=X_processed, y=y)

        # On met à jour les predicteurs car le resampling (notamment nearmiss) peut crééer des features de variance nulle
        null_var = check_variances(X_balanced, raise_error=False, verbose=False)
        if null_var:
            predictors = [f for f in X_balanced.columns if f not in null_var]
        return X_balanced[predictors]

    def run_top_10_correlations(self, X, corr_matrix, verbose=True):

        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=f"top_10_correlations",
        ) as run:
            self._run_id.append(run.info.run_id)
            mlflow.log_params({"corr_coef": self.corr_coef})

            # On loggue le dataset
            self.create_mlflow_dataset()
            self.track_dataset()

            # On plotte les features les plus corrélées entre elles
            max_features = 10
            title = f"Matrice des corrélations {self.corr_coef.upper()} - Top {max_features} des plus hautes corrélations"
            subtitle = f"pipe : {self.get_pipe_str()}\nRéquilibrage : {self.meta.balance.upper()}\nExp : {self.meta._name}"
            top_corr = plot_top_correlations(
                corr_matrix,
                max_n_features=max_features,
                title=title,
                subtitle=subtitle,
                verbose=verbose,
            )
            # On sauvegarde la figure (la précédente sera écrasée, c'est volontarire)
            filepath = os.path.join(self.output_dir, "top_correlations.png")
            top_corr.savefig(filepath, bbox_inches="tight")
            # On loggue dans mlflow
            mlflow.log_artifact(filepath)

    def run_features_corr_above(self, X, corr_matrix, verbose=True):
        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=f"correlations_above_{self.threshold_corr}",
        ) as run:
            self._run_id.append(run.info.run_id)
            mlflow.log_params(
                {"corr_coef": self.corr_coef, "threshold_corr": self.threshold_corr}
            )

            # On loggue le dataset
            self.create_mlflow_dataset()
            self.track_dataset()
            # On récupère la liste des features qui ont une corrélation supérieure au seuil (en valeur absolue)
            features_correlated_above_threshold = get_features_correlated_above(
                corr_matrix, self.threshold_corr, verbose=verbose
            )
            self._features_correlated_above_threshold = (
                features_correlated_above_threshold
            )

            # On loggue la liste des features préselectionnées comme hautement corrélées 2 à 2 dans mlflow
            dic_features_correlated_above_threshold = {
                "features_correlated_above_threshold": self._features_correlated_above_threshold
            }
            mlflow.log_dict(
                dic_features_correlated_above_threshold,
                "features_correlated_above_threshold.json",
            )
        return features_correlated_above_threshold

    def run_redondant_features(self, X, verbose=True):
        run_name = f"redondant_features_{self.threshold_dist}"
        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=run_name,
        ) as run:
            self._run_id.append(run.info.run_id)
            # On loggue le dataset et les paramètres
            self.create_mlflow_dataset()
            self.track_dataset()
            mlflow.log_params(
                {
                    "corr_coef": self.corr_coef,
                    "threshold_corr": self.threshold_corr,
                    "threshold_dist": self.threshold_dist,
                }
            )
            linkage_matrix = build_linkage_matrix(
                X,
                self._features_correlated_above_threshold,
                corr_coef=self.corr_coef,
                method=self.agglo_method,
            )
            all_clusters_col, features_to_drop = cluster_features_from_linkage_matrix(
                X,
                linkage_matrix,
                features=self._features_correlated_above_threshold,
                threshold_pct=self.threshold_dist,
                verbose=verbose,
            )
            self._features_to_drop = features_to_drop

            # On logue les features à supprimer
            dic_features_to_drop = {"features_to_drop": features_to_drop}
            mlflow.log_dict(dic_features_to_drop, "features_to_drop.json")

            # On plotte le dendrogramme
            subtitle = (
                f"Métrique={self.metric}, seuil distance : {self.threshold_dist:.0%}"
            )
            balance_str = f" - {self.meta.balance.upper()}"
            if self.meta.balance == "none":
                balance_str = ""
            subtitle = (
                subtitle
                + f"\nPipe : {self.get_pipe_str()}{balance_str}\nExp : {self.meta._name}"
            )
            dendro = plot_dendro(
                linkage_matrix,
                features=self._features_correlated_above_threshold,
                corr_coef=self.corr_coef,
                threshold_dist=self.threshold_dist,
                dropped_features=features_to_drop,
                subtitle=subtitle,
                verbose=verbose,
            )
            # On sauvegarde la figure et on la loggue dans mlflow
            filepath = os.path.join(self.output_dir, "features_clustering.png")
            dendro.savefig(filepath, bbox_inches="tight")
            mlflow.log_artifact(filepath)

    def drop_redondant_features(self):
        self.X = self.X.drop(self._features_to_drop, axis=1)
        print(f"{len(self._features_to_drop)} features supprimées")
        memory_consumed_mb = get_memory_consumed(self.X, verbose=False)
        print(
            f"Nouvelle forme de X : {self.X.shape}, conso mémoire : {memory_consumed_mb:.0f} Mo"
        )


class ExpEvalLogreg(ExpMlFlow):
    def __init__(
        self,
        debug=False,
    ):
        super().__init__(task="eval", model_name="logreg", debug=debug)
        self.all_params = None
        self.model = None
        # self._default_model = LogisticRegression()
        self._param_names_of_model = LogisticRegression().get_param_names()
        self.device = "cuda"
        self.pipe_preprocess = None

        self.meta.balance = "none"
        self.metric = "auc"
        self.max_iter = 1_000
        self.tol = 0.000_1
        self.X = None
        self.y = None
        self._run_id = None
        # self.threshold_prob = threshold_prob

    def init_all_params(self, params):
        dic_params = deepcopy(params)
        # [TODO ? ] plutôt éliminer des params ceux dont les valeurs sont à 'none' ou à None dans le dico final ?
        if not "k_neighbors" in params.keys():
            dic_params["k_neighbors"] = 0
        if not "balance" in params.keys():
            dic_params["balance"] = "none"
        if not "threshold_prob" in params.keys():
            dic_params["threshold_prob"] = 0.5
        if not "penalty" in params.keys():
            dic_params["penalty"] = "l2"
        # Nos paramètre sont issus de recheche Optuna.
        # dans optuna il faut "none" as string (pas None) et dans le fonctionnement normal il faut None (ou param absent)
        if "class_weight" in params.keys() and params["class_weight"] == "none":
            dic_params["class_weight"] = None
        if not "verbose" in params.keys():
            dic_params["verbose"] = cuml.common.logger.level_error
        return dic_params

    def init_run_name(self, all_params):
        run_name = f"{self.get_run_count() + 1:03}"
        if all_params["balance"] == "none":
            run_name = run_name + f"_unbalanced"
        else:
            run_name = run_name + f"_{all_params['balance']}"
        return run_name

    def init_run_tags(self, all_params, verbose=False):
        params_to_tag_from_all_params = [
            "balance",
            "k_neighbors",
            "penalty",
            "class_weight",
            "search_experiment",
        ]
        tags = {
            k: v for k, v in all_params.items() if k in params_to_tag_from_all_params
        }
        tags["task"] = self.meta.task
        if verbose:
            print("Tags du run :")
            print(tags)
        return tags

    def read_data(self, verbose=True):
        super().read_data(verbose)
        self.create_mlflow_dataset()

    def run(self, params, n_folds=5, verbose=True):
        if not self._mlflow_id:
            print("Créez l'expérience mlflow avant de lancer un run")
            return
        all_params = self.init_all_params(params)
        run_name = self.init_run_name(all_params)
        run_tags = self.init_run_tags(all_params)

        if verbose:
            self.print_params(all_params)
            print("run_name :", run_name)
        print(f"Evaluation sur {self.device}...")
        t0_score = time.time()
        with mlflow.start_run(
            experiment_id=self._mlflow_id,
            run_name=run_name,
        ) as run:
            self._run_id = run.info.run_id

            # On ajoute des tags au run pour pouvoir les sélectionner grâce à eux :
            mlflow.set_tags(run_tags)

            # On loggue les paramètres dans mlflow
            mlflow.log_params(all_params)
            params_of_model = self.get_params_of_model(all_params)
            model = LogisticRegression(**params_of_model)

            # On logue le modèle (non_fitté)
            # mlflow.sklearn.log_model(model, "saved_models")

            # Pour loguer le dataset
            if all_params["balance"] == "none":
                context = "Unbalanced"
            else:
                context = all_params["balance"]
            self.track_dataset(context=context)

            # On logue les paramètres du modèle sous forme de json
            dic_params_model = {"dic_params_model": params_of_model}
            mlflow.log_dict(dic_params_model, "dic_params_model.json")
            # On logue tous les paramètres sous forme de json
            dic_all_params = {"dic_all_params": all_params}
            mlflow.log_dict(dic_all_params, "dic_all_params.json")

            train_scores, val_scores = cuml_cross_evaluate(
                X_train_and_val=self.X,
                y_train_and_val=self.y,
                pipe_preprocess=self.pipe_preprocess,
                cuml_model=model,
                balance=all_params["balance"],
                k_neighbors=all_params["k_neighbors"],
                n_folds=n_folds,
                threshold_prob=all_params["threshold_prob"],
                train_scores=True,
            )
            print(
                "Durée du scoring en cross évaluation :",
                format_time(time.time() - t0_score),
            )
            # On ajoute un préfixe devant les scores pour les différencier dans mlflow
            train_scores_prefix = {f"train.{k}": v for k, v in train_scores.items()}
            val_scores_prefix = {
                f"val.{k}": v for k, v in val_scores.items() if k != "fit_time"
            }

            mean_train_scores = {
                k: np.mean(v) for (k, v) in train_scores_prefix.items()
            }
            mean_val_scores = {k: np.mean(v) for (k, v) in val_scores_prefix.items()}
            # On logue les scores moyens
            mlflow.log_metrics(mean_train_scores)
            mlflow.log_metrics(mean_val_scores)

            title = (
                f"Scores moyens moyen sur {n_folds} folds - {model.__class__.__name__}"
            )
            subtitle = f"Mlflow : {self.meta._name}, {run_name}"
            if (
                "search_experiment" in all_params.keys()
                and all_params["search_experiment"]
            ):
                subtitle = (
                    subtitle + f"\nSrc params : {all_params['search_experiment']}"
                )

            # On plotte les scores d'évaluation
            fig = plot_evaluation_scores(
                train_scores=train_scores,
                val_scores=val_scores,
                title=title,
                subtitle=subtitle,
                verbose=verbose,
            )
            # On logue le plot dans mlflow :
            filepath = os.path.join(self.output_dir, "eval_scores.png")
            fig.savefig(filepath)
            # On loggue dans mlflow
            mlflow.log_artifact(filepath)

            # On plotte la matrice de confusion moyenne (recall) sur le jeu de valdation
            title = f"Recall moyen sur {n_folds} folds - {model.__class__.__name__}"
            fig = plot_recall_mean(
                tn=mean_val_scores["val.tn"],
                fp=mean_val_scores["val.fp"],
                fn=mean_val_scores["val.fn"],
                tp=mean_val_scores["val.tp"],
                title=title,
                subtitle=subtitle,
                verbose=verbose,
            )
            # On sauvegarde la figure et on la logue dans mlflow
            filepath = os.path.join(self.output_dir, "recall_mean_plot.png")
            fig.savefig(filepath, bbox_inches="tight")
            # On loggue dans mlflow
            mlflow.log_artifact(filepath)
            print("Durée totale :", format_time(time.time() - t0_score))
        return

    def get_params_of_model(self, all_params):
        return {k: v for k, v in all_params.items() if k in self._param_names_of_model}

    def print_params(self, all_params):
        params_model = self.get_params_of_model(all_params)
        params_other = {
            k: v for k, v in all_params.items() if k not in params_model.keys()
        }
        print(f"Paramètres du modèle :")
        for k, v in params_model.items():
            print(f"\t{k} : {v}")
        print("Autres paramètres :")
        for k, v in params_other.items():
            print(f"\t{k} : {v}")


def get_run_id_from_name(experiment_id, run_name):
    runs = mlflow.search_runs(experiment_ids=experiment_id)
    run_id_serie = runs.loc[runs["tags.mlflow.runName"] == run_name, "run_id"]
    if run_id_serie is not None:
        return run_id_serie.values[0]
    return None


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


def del_artifacts_from_mlflow(artifacts_rootdir="mlflow_artifacts", verbose=True):

    existing_experiments = mlflow.search_experiments(
        view_type=mlflow.entities.ViewType.ACTIVE_ONLY
    )
    existing_experiment_names = [experiment.name for experiment in existing_experiments]
    # print("existing_experiment_names")
    # print(existing_experiment_names)

    existing_artifact_dirs = [
        name
        for name in os.listdir(artifacts_rootdir)
        if os.path.isdir(os.path.join(artifacts_rootdir, name))
    ]
    if "download" in existing_artifact_dirs:
        existing_artifact_dirs.remove("download")
    # print("existing_artifact_dirs")
    # print(existing_artifact_dirs)

    artifact_dirs_to_delete = [
        os.path.join(artifacts_rootdir, name)
        for name in existing_artifact_dirs
        if name not in existing_experiment_names
    ]
    if artifact_dirs_to_delete:
        if verbose:
            print(
                f"{len(artifact_dirs_to_delete)} répertoires contenant des artifacts seront supprimés du disque dur :"
            )
            print(artifact_dirs_to_delete)

        confirmed = ask_confirmation(verbose=verbose)
        if confirmed:
            for dir_name in artifact_dirs_to_delete:
                shutil.rmtree(dir_name)
            return artifact_dirs_to_delete
    else:
        if verbose:
            print("Aucun artifact à supprimer")

    return []


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
        exp.name
        for exp in all_experiments
        if exp.name.startswith("search_") or exp.name.startswith("debug_search_")
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
