import numpy as np
import random
import logging
import io
import re
import requests
import os
import gc
import time
import joblib
from joblib import Parallel, delayed
from copy import deepcopy
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
from scipy import special
import cudf
import cuml
from cuml.linear_model import LogisticRegression
from bokbokbok.loss_functions.classification import WeightedCrossEntropyLoss
from bokbokbok.eval_metrics.classification import WeightedCrossEntropyMetric
from bokbokbok.utils import clip_sigmoid

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
from src.p7_metric import (
    cu_pred_prob_to_binary,
    pd_pred_prob_to_binary,
    penalize_f1,
    penalize_business_gain,
    make_objective_weighted_logloss,
)
from src.p7_evaluate import (
    cuml_cross_validate,
    cuml_cross_evaluate,
    balance_smote,
    balance_nearmiss,
    lgb_validate_1_fold,
    pre_process_1_fold,
)
from src.p7_evaluate import CuDummyClassifier
from src.p7_preprocess import train_test_split_nan
from src.p7_tracking import ExpMetaData, ExpMlFlow, ExpSearch
from src.p7_metric import (
    logloss_objective,
    logloss_metric,
    feval_auc,
    make_feval_business_gain,
)


class SearchCuDummy(ExpSearch):
    def __init__(self, metric="auc", direction="maximize", debug=False):
        super().__init__(
            model_name="dummy", metric=metric, direction=direction, debug=debug
        )
        self.device = "cuda"
        # self.sampler = optuna.samplers.TPESampler(seed=VAL_SEED)
        # Liste des paramètres qu'on veut voir affichés dans le parrallel plot fait par optuna
        self._params_to_plot_in_parallel = []

    def objective(self, trial):
        with mlflow.start_run(
            experiment_id=self._mlflow_id, run_name=f"{trial.number}", nested=True
        ):

            threshold_prob = trial.suggest_float("threshold_prob", 0.0, 1.0, log=False)

            # Si la métric à optimiser est l'auc, on ne fixe pas les graines de hasard
            # car l'auc ne dépendant pas du seul de proba, nous aurons toujours la même auc
            # Par contre pour les autres métriques on peut fixer les graines de hasard car le seuil de proba modifie la métrique résultante
            if self.metric == "auc":
                clf = CuDummyClassifier(random_state=None)
            else:
                clf = CuDummyClassifier(random_state=VAL_SEED)
            all_params = {"threshold_prob": threshold_prob}

            # Si on n'a pas d'oversampling ni d'undersampling, on fait une cross validation avec 5 folds,
            # Si on a de l'oversampling ou de l'undersampling on ne fait que 2 folds pour diminuer les temps de calcul
            if self.meta.balance == "none":
                n_folds = 5
                k_neighbors = 0
            else:
                n_folds = 2
                k_neighbors = trial.suggest_int("k_neighbors", 3, 6)
                all_params["k_neighbors"] = k_neighbors

            scores = cuml_cross_evaluate(
                X_train_and_val=self.X,
                y_train_and_val=self.y,
                pipe_preprocess=self.pipe_preprocess,
                cuml_model=clf,
                balance=self.meta.balance,
                k_neighbors=k_neighbors,
                n_folds=n_folds,
                threshold_prob=threshold_prob,
                train_scores=False,
            )

            # mean_scores = scores.mean(axis=0)
            mean_scores = {k: np.mean(v) for (k, v) in scores.items()}

            # Minimise la log loss : ll = log_loss(test_y , y_predlr)

            # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
            # on personnalise pour enregistrer toutes les métriques
            trial.set_user_attr("mean_scores", mean_scores)

            # Fin d'un trial, on loggue dans mlflow
            mlflow.log_metrics(mean_scores)

            mlflow.log_params(all_params)
            # On ne logue le dataset que dans le best_run
            # self.track_dataset()

        return mean_scores[self.metric]

    def run_logreg_trials(self, n_trials=20, verbose=True):
        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials...")

        self._study.optimize(self.objective, n_trials=n_trials, gc_after_trial=True)
        if verbose:
            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))

    # La parallélisation ne fonctionne pas si CUDA car pas assez de mémoire
    # ou tout simplement allocation sur CUDA en parallel ne fonctionne pas bien (alloue plus que nécessaire).
    # On peut surveiller en temps réel la consommation de VRAM avec nvidia-smi -l 1 (crée une loop pour rafraichir toutes les 1 secondes)
    # Le multithreads réglerait peut-être le problème de l'allocation VRAM mais MLFlow n'est pas thread-safe.
    def optimize(self, n_trials=10, verbose=True):
        # Si le nombre de trials est grand, on ne les loggue pas à l'écran, à la place on affiche uniquement les warnings
        if n_trials > 20:
            self.optuna_verbosity = optuna.logging.WARNING
        else:
            self.optuna_verbosity = optuna.logging.INFO
        optuna.logging.set_verbosity(self.optuna_verbosity)

        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials sur CUDA...")

        # self.run_logreg_trials(n_trials=n_trials, n_jobs=1, verbose=False)
        self._study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )

        if verbose:
            """print(
                f"Meilleure combinaison : trial {self._study.best_trial.number}, {self.metric} : {self._study.best_trial.value}"
            )
            print(f"Paramètres : {self._study.best_trial.params}")"""

            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))
        self.create_best_run(verbose=verbose)


class SearchLogReg(ExpSearch):
    def __init__(self, metric="auc", direction="maximize", debug=False):
        super().__init__(
            model_name="logreg",
            metric=metric,
            direction=direction,
            debug=debug,
            loss="binary",
        )
        self.meta.description2 = (
            f"Optimisation de {metric}, Fonction de perte : 'binary'"
        )
        self.device = "cuda"
        # Comme CUDA est utilisé, on ne parallélise pas, on peut donc fiwxer la graine du sampler
        self.sampler = optuna.samplers.TPESampler(seed=VAL_SEED)
        # Liste des paramètres qu'on veut voir affichés dans le parrallel plot fait par optuna
        self._params_to_plot_in_parallel = [
            "C",
            "penalty",
            "class_weight",
            "threshold_prob",
            "fit_intercept",
        ]

    def objective(self, trial):

        with mlflow.start_run(
            experiment_id=self._mlflow_id, run_name=f"{trial.number}", nested=True
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
                "tol": 1e-4,
                "fit_intercept": fit_intercept,
            }
            # Petit bug pour le cuml LogisticRegression, on ne peut pas directement assigner class_weight
            if class_weight == "balanced":
                clf_params["class_weight"] = "balanced"

            # Max_iter par défaut est 1_000. Si une régularisation l1 est appliquée, dans ces conditions,
            # L'algorithme ne pourra pas converger. Soit on augmente max_iter, (voire la tol) soit on scale les features binaires dans le pipe.
            if penalty == "l1" or penalty == "elasticnet":
                clf_params["max_iter"] = 10_000
                clf_params["tol"] = 1e-3
                if penalty == "elasticnet":
                    clf_params["l1_ratio"] = l1_ratio

            # On concentre la recherche du seuil autour de 0.5 avec la distribution Beta
            # threshold_prob = self.suggest_beta("threshold_prob", low=0.05, high=0.95)
            threshold_prob = trial.suggest_float("threshold_prob", 0.0, 1.0, log=False)

            # La verbosité permet de désactiver le warning : QWL-QN stopped, because the line search failed to advance (step delta = 0.000000)
            clf = LogisticRegression(
                **clf_params, verbose=cuml.common.logger.level_error
            )
            """scores = cuml_cross_validate(
                self.X,
                self.y,
                self.pipe_preprocess,
                clf,
                threshold_prob=threshold_prob,
            )"""

            all_params = {k: v for (k, v) in clf_params.items()}
            all_params["threshold_prob"] = threshold_prob

            # Si on n'a pas d'oversampling ni d'undersampling, on fait une cross validation avec 5 folds,
            # Si on a de l'oversampling ou de l'undersampling on ne fait que 2 folds pour diminuer les temps de calcul
            if self.meta.balance == "none":
                n_folds = 5
                k_neighbors = 0
            else:
                n_folds = 2
                k_neighbors = trial.suggest_int("k_neighbors", 3, 6)
                all_params["k_neighbors"] = k_neighbors

            scores = cuml_cross_evaluate(
                X_train_and_val=self.X,
                y_train_and_val=self.y,
                pipe_preprocess=self.pipe_preprocess,
                cuml_model=clf,
                balance=self.meta.balance,
                k_neighbors=k_neighbors,
                n_folds=n_folds,
                threshold_prob=threshold_prob,
                train_scores=False,
            )

            # mean_scores = scores.mean(axis=0)
            mean_scores = {k: np.mean(v) for (k, v) in scores.items()}

            # Minimise la log loss : ll = log_loss(test_y , y_predlr)

            # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
            # on personnalise pour enregistrer toutes les métriques
            trial.set_user_attr("mean_scores", mean_scores)

            # Fin d'un trial, on loggue dans mlflow
            mlflow.log_metrics(mean_scores)
            """all_params = {k: v for (k, v) in clf_params.items()}
            all_params["threshold_prob"] = threshold_prob"""

            mlflow.log_params(all_params)
            # On ne logue le dataset que dans le best_run
            # self.track_dataset()

        return mean_scores[self.metric]

    def run_logreg_trials(self, n_trials=20, n_jobs=1, verbose=True):
        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials...")

        self._study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )
        if verbose:
            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))

    # La parallélisation ne fonctionne pas si CUDA car pas assez de mémoire
    # ou tout simplement allocation sur CUDA en parallel ne fonctionne pas bien (alloue plus que nécessaire).
    # On peut surveiller en temps réel la consommation de VRAM avec nvidia-smi -l 1 (crée une loop pour rafraichir toutes les 1 secondes)
    # Le multithreads réglerait peut-être le problème de l'allocation VRAM mais MLFlow n'est pas thread-safe.
    def run(self, n_trials=10, verbose=True):
        # Si le nombre de trials est grand, on ne les loggue pas à l'écran, à la place on affiche uniquement les warnings
        if n_trials > 20:
            self.optuna_verbosity = optuna.logging.WARNING
        else:
            self.optuna_verbosity = optuna.logging.INFO
        optuna.logging.set_verbosity(self.optuna_verbosity)

        t0 = time.time()
        if verbose:
            print(f"Optimisation {n_trials} trials sur CUDA...")

        # self.run_logreg_trials(n_trials=n_trials, n_jobs=1, verbose=False)
        self._study.optimize(
            self.objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True
        )

        if verbose:
            """print(
                f"Meilleure combinaison : trial {self._study.best_trial.number}, {self.metric} : {self._study.best_trial.value}"
            )
            print(f"Paramètres : {self._study.best_trial.params}")"""

            print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))
        self.create_best_run(verbose=verbose)

    def get_model(self):
        params_of_model = self.get_params_of_model()
        return LogisticRegression(**params_of_model)


# [TODO] refacto car hérite de ExpSearch au lieu de ExperimentSearch
class SearchLgb(ExpSearch):
    def __init__(self, metric="auc", direction="maximize", loss="binary", debug=False):
        super().__init__(
            model_name="lgbm",
            metric=metric,
            direction=direction,
            debug=debug,
            loss=loss,
        )
        self.meta.balance = "none"
        self.meta.description2 = f"Optimisation de {metric}, Fonction de perte : {loss}"
        self.device = "cpu"
        # Finalement il n'est pas intéressant de paralléliser, on fixe donc la graine du sampler
        self.sampler = optuna.samplers.TPESampler(VAL_SEED)
        # Grand maximum en séquentiel
        self.lgb_n_threads = 12
        # Turn off optuna log notes.
        # optuna.logging.set_verbosity(optuna.logging.WARN)
        # Liste des paramètres qu'on veut voir affichés dans le parrallel plot fait par optuna
        self._params_to_plot_in_parallel = [
            "n_estimators",
            "learning_rate",
            "lambda_l1",
            "lambda_l2",
            "num_leaves",
            "feature_fraction",
            # "is_unbalance",
            "scale_pos_weight",
            "threshold_prob",
        ]
        self.min_delta = 0.001
        # self.sampler = optuna.samplers.TPESampler(VAL_SEED)
        self.n_folds = 4

        # Assure la reproductibilité de l'entraînement lgbm si plusieurs thrreads (ralentit)
        self.deterministic = False

    # On ne teste pas boosting_type dart car pas de early_stopping et c'est trop long
    # Pas de validation croisée car c'est trop long
    def objective(self, trial):
        with mlflow.start_run(
            experiment_id=self._mlflow_id, run_name=f"{trial.number}", nested=True
        ):
            threshold_prob = trial.suggest_float("threshold_prob", 0.0, 1.0, log=False)
            learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5, log=True)
            max_bin = trial.suggest_int("max_bin", 128, 512, step=32)
            n_estimators = trial.suggest_int("n_estimators", 40, 400, step=20)
            num_leaves = trial.suggest_int("num_leaves", 8, 256)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
            # k_neighbors = trial.suggest_int("k_neighbors", 3, 6)
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0)
            lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
            lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
            # Permet d'utiliser un sous-ensemble des features sélectionnées aléatoirement
            # et ainsi de varier les arbres. Aide contre surfit et diminue temps de fit
            feature_fraction = trial.suggest_float(
                "feature_fraction", 0.4, 1.0, step=0.1
            )
            # Idem feature_fraction mais pour les lignes
            bagging_fraction = trial.suggest_float(
                "bagging_fraction", 0.4, 1.0, step=0.1
            )
            bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
            alpha = trial.suggest_float("alpha", 1.0, 20.0)

            # On définit la métrique sur laquelle va s'exercer le early_stopping et le pruning
            if self.metric in ["auc", "business_gain"]:
                metrics = [self.metric]
            else:
                raise ValueError(
                    f"La métrique '{self.metric}' n'est pas autorisée. Métriques possibles à optimiser : ['auc', 'business_gain']"
                )

            # On définit le callback de pruning pour interrompre l'essai optuna s'il n'est pas prometteur
            # et le callback de early_stopping pour interrompre l'entraînement si la métrique d'évaluation n'évolue plus
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, self.metric
            )
            eval_results = {}
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=10, min_delta=self.min_delta, verbose=True
                ),
                pruning_callback,
                # lgb.log_evaluation(period=5),
                lgb.record_evaluation(eval_results),
            ]

            # Si on utilise la fonction de perte binary_logloss
            if self.loss == "binary":
                lgb_objective = "binary"
                if self.metric == "auc":
                    feval = None
                else:
                    # La fonction de perte étant "buit-in", le modèle renvoie des probas et non des logits
                    feval_business = make_feval_business_gain(
                        threshold_prob, preds_are_logits=False
                    )
                    feval = [feval_business]

                alpha = 1.0

            # Si on utilise une fonction de perte personnalisée qui pondère la binary_cross_entropy
            elif self.loss == "weighted_logloss":
                # alpha < 1.0 réduit le nombre de faux négatifs, 1 équivaut à loss="binary" (à condition de boost_from_average=False)
                # et alpha > 1 réduirait le nombre de faux positifs
                # alpha = trial.suggest_float("alpha", 1.0, 20.0)
                lgb_objective = make_objective_weighted_logloss(alpha)
                if self.metric == "auc":
                    feval = [feval_auc]
                else:
                    # La fonction de perte étant personnalisée, le modèle renvoie des logits et non des probas
                    feval_business = make_feval_business_gain(
                        threshold_prob, preds_are_logits=True
                    )
                    feval = [feval_business]

            # Si on a de l'oversampling ou de l'undersampling, on recherche le nombre de voisins,
            # ainsi que le ratio classe minoritaire / classe majoritaire
            if self.meta.balance == "none":
                k_neighbors = 0
                sampling_strategy = 0
            else:
                k_neighbors = trial.suggest_int("k_neighbors", 3, 6)
                sampling_strategy = trial.suggest_float("sampling_strategy", 0.1, 1.0)

            lgb_params = {
                "force_col_wise": True,
                "boosting_type": "gbdt",
                # "boost_from_average": False,
                "deterministic": self.deterministic,
                "objective": lgb_objective,
                "metrics": metrics,
                "scale_pos_weight": scale_pos_weight,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "min_child_samples": min_child_samples,
                "learning_rate": learning_rate,
                "max_bin": max_bin,
                "num_threads": self.lgb_n_threads,
                "verbosity": -1,
                "verbose_eval": False,
            }

            # Paramètres utilisés dans le train et dans le scoring
            other_params = {
                "n_estimators": n_estimators,
                "feval": feval,
                "callbacks": callbacks,
                "loss": self.loss,
                "threshold_prob": threshold_prob,
            }
            all_params = {
                k: v
                for k, v in lgb_params.items()
                if k
                not in [
                    "random_seed",
                    "verbosity",
                    "verbose",
                    "verbose_eval",
                    "objective",
                    "metrics",
                    "num_threads",
                ]
            }
            all_params["n_estimators"] = n_estimators
            all_params["loss"] = self.loss
            all_params["alpha"] = alpha
            all_params["balance"] = self.meta.balance
            all_params["k_neigbors"] = k_neighbors
            all_params["sampling_strategy"] = sampling_strategy
            mlflow.log_params(all_params)

            predictors = [
                f for f in self.X.columns if f not in ["SK_ID_CURR", "TARGET"]
            ]

            # Si on utilise pas la validation croisée (temps de calculs trop longs)
            if self.n_folds == 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    self.X[predictors],
                    self.y,
                    stratify=self.y,
                    test_size=0.25,
                    random_state=VAL_SEED,
                )
                # Conversion des indices en indices positionnels relatifs à self.X
                train_idx = self.X.index.get_indexer(X_train.index)
                valid_idx = self.X.index.get_indexer(X_val.index)
                folds_list = [(train_idx, valid_idx)]

            # Si on utilise la validation croisée:
            else:
                folds = StratifiedKFold(
                    n_splits=self.n_folds,
                    shuffle=True,
                    random_state=VAL_SEED,
                )
                folds_list = [fold for fold in folds.split(self.X[predictors], self.y)]

            all_scores = {
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

            for train_idx, valid_idx in folds_list:
                X_train = self.X[predictors].iloc[train_idx]
                X_val = self.X[predictors].iloc[valid_idx]
                y_train = self.y.iloc[train_idx]
                y_val = self.y.iloc[valid_idx]

                X_train_balanced, y_train_balanced, X_val_processed = (
                    pre_process_1_fold(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        pipe_preprocess=self.pipe_preprocess,
                        balance=self.meta.balance,
                        k_neighbors=k_neighbors,
                        sampling_strategy=sampling_strategy,
                    )
                )

                fold_scores = lgb_validate_1_fold(
                    X_train=X_train_balanced,
                    y_train=y_train_balanced,
                    X_test=X_val_processed,
                    y_test=y_val,
                    model_params=lgb_params,
                    other_params=other_params,
                )
                for k in all_scores.keys():
                    all_scores[k].append(fold_scores[k])

                del X_train
                del X_train_balanced
                del y_train
                del y_train_balanced
                del X_val
                del X_val_processed
                del y_val
                del fold_scores

                gc.collect()

            # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
            # On utilise set_user_attr qui permet de personnaliser ce qu'on veut logguer dans Optuna
            mean_scores = {k: np.mean(v) for (k, v) in all_scores.items()}
            trial.set_user_attr("mean_scores", mean_scores)

            # On loggue les scores moyens dans mlflow
            mlflow.log_metrics(mean_scores)

        return mean_scores[self.metric]

    def objective_old(self, trial):
        with mlflow.start_run(
            experiment_id=self._mlflow_id, run_name=f"{trial.number}", nested=True
        ):
            threshold_prob = trial.suggest_float("threshold_prob", 0.0, 1.0, log=False)
            learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.5, log=True)
            max_bin = trial.suggest_int("max_bin", 128, 512, step=32)
            n_estimators = trial.suggest_int("n_estimators", 40, 400, step=20)

            # On définit la métrique sur laquelle va s'exercer le early_stopping et le pruning
            if self.metric in ["auc", "business_gain"]:
                metrics = [self.metric]
            else:
                raise ValueError(
                    f"La métrique '{self.metric}' n'est pas autorisée. Métriques possibles à optimiser : ['auc', 'business_gain']"
                )

            """# On définit le callback de pruning pour interrompre l'essai optuna s'il n'est pas prometteur
            # et le callback de early_stopping pour interrompre l'entraînement si la métrique d'évaluation n'évolue plus
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial, self.metric
            )"""
            eval_results = {}
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=10, min_delta=self.min_delta, verbose=True
                ),
                # pruning_callback,
                # lgb.log_evaluation(period=5),
                lgb.record_evaluation(eval_results),
            ]

            # Si on utilise la fonction de perte binary_logloss
            if self.loss == "binary":
                lgb_objective = "binary"
                if self.metric == "auc":
                    feval = None
                else:
                    # La fonction de perte étant "buit-in", le modèle renvoie des probas et non des logits
                    feval_business = make_feval_business_gain(
                        threshold_prob, preds_are_logits=False
                    )
                    feval = [feval_business]

                alpha = 1.0
                num_leaves = trial.suggest_int("num_leaves", 8, 256)
                min_child_samples = trial.suggest_int("min_child_samples", 5, 100)

            # Si on utilise une fonction de perte personnalisée qui pondère la binary_cross_entropy
            elif self.loss == "weighted_logloss":
                # alpha < 1.0 réduit le nombre de faux négatifs, 1 équivaut à loss="binary" (à condition de boost_from_average=False)
                # et alpha > 1 réduirait le nombre de faux positifs
                alpha = trial.suggest_float("alpha", 1.0, 20.0)
                lgb_objective = make_objective_weighted_logloss(alpha)
                if self.metric == "auc":
                    feval = [feval_auc]
                else:
                    # La fonction de perte étant personnalisée, le modèle renvoie des logits et non des probas
                    feval_business = make_feval_business_gain(
                        threshold_prob, preds_are_logits=True
                    )
                    feval = [feval_business]
                # On réduit l'espace de recherche si alpha < 1, car le modèle ne peut plus fitter sur des grandes valeurs
                # Fonctionne mais provoque le warning [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
                num_leaves = trial.suggest_int("num_leaves", 8, 64)
                min_child_samples = trial.suggest_int("min_child_samples", 5, 20)

            # équivallent class_weight mais pour classfification binaire
            # is_unbalance = trial.suggest_categorical("is_unbalance", ["true", "false"])
            scale_pos_weight = trial.suggest_float("scale_pos_weight", 1.0, 10.0)
            lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
            lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
            # Permet d'utiliser un sous-ensemble des features sélectionnées aléatoirement
            # et ainsi de varier les arbres. Aide contre surfit et diminue temps de fit
            feature_fraction = trial.suggest_float(
                "feature_fraction", 0.4, 1.0, step=0.1
            )
            # Idem feature_fraction mais pour les lignes
            bagging_fraction = trial.suggest_float(
                "bagging_fraction", 0.4, 1.0, step=0.1
            )
            bagging_freq = trial.suggest_int("bagging_freq", 1, 7)

            # Si on a de l'oversampling ou de l'undersampling, on recherche le nombre de voisins,
            # On ne recherche pas la sampling_strategy car on n'a pas la puissance de calcul
            if self.meta.balance != "none":
                k_neighbors = trial.suggest_int("k_neighbors", 3, 6)
            else:
                k_neighbors = 0

            lgb_params = {
                "force_col_wise": True,
                "boosting_type": "gbdt",
                # "boost_from_average": False,
                # "deterministic": True,
                "objective": lgb_objective,
                "metrics": metrics,
                "scale_pos_weight": scale_pos_weight,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
                "num_leaves": num_leaves,
                "feature_fraction": feature_fraction,
                "bagging_fraction": bagging_fraction,
                "bagging_freq": bagging_freq,
                "min_child_samples": min_child_samples,
                "learning_rate": learning_rate,
                "max_bin": max_bin,
                "num_threads": self.lgb_n_threads,
                "verbosity": -1,
                "verbose_eval": False,
            }
            all_params = {
                k: v
                for k, v in lgb_params.items()
                if k
                not in [
                    "random_seed",
                    "verbosity",
                    "verbose",
                    "verbose_eval",
                    "objective",
                    "metrics",
                    "num_threads",
                ]
            }
            all_params["n_estimators"] = n_estimators
            all_params["loss"] = self.loss
            all_params["alpha"] = alpha
            all_params["balance"] = self.meta.balance
            all_params["k_neigbors"] = k_neighbors
            mlflow.log_params(all_params)

            predictors = [
                f for f in self.X.columns if f not in ["SK_ID_CURR", "TARGET"]
            ]

            # On n'utilise pas de validation croisée, les temps de calculs sont trop longs
            X_train, X_val, y_train, y_val = train_test_split(
                self.X[predictors],
                self.y,
                stratify=self.y,
                test_size=0.25,
                random_state=VAL_SEED,
            )

            # Eventuel pipeline de pré-traitement
            if self.pipe_preprocess:
                pipe = deepcopy(self.pipe_preprocess)
                X_train_processed = pipe.fit_transform(X_train)
                X_val_processed = pipe.transform(X_val)
            else:
                X_train_processed = X_train
                X_val_processed = X_val

            # Eventuel rééquilibrage avec SMOTE ou NearMiss
            if self.meta.balance == "none":
                X_train_balanced = X_train_processed
                y_train_balanced = y_train
            elif self.meta.balance == "smote":
                # Si la sampling_strategy n'a pas été définie, on utilise par défaut 0.7
                # Cela ne rééquilibre pas à 50% (trop long, trop de ram) mais c'est suffisant.
                if self.meta.sampling_strategy == 0:
                    self.meta.sampling_strategy = 0.7
                X_train_balanced, y_train_balanced = balance_smote(
                    X_train_processed,
                    y_train,
                    k_neighbors=k_neighbors,
                    sampling_strategy=self.meta.sampling_strategy,
                    random_state=VAL_SEED,
                    verbose=False,
                )
            elif self.meta.balance == "nearmiss":
                X_train_balanced, y_train_balanced, features_null_var = (
                    balance_nearmiss(
                        X_train_processed,
                        y_train,
                        k_neighbors=k_neighbors,
                        verbose=False,
                    )
                )
            else:
                print(
                    f"{self.meta.balance} est une valeur incorrecte pour balance. Les valeurs possibles sont 'none', 'smote' ou 'nearmiss'"
                )

            # On transforme en dataset lgb
            lgb_train = lgb.Dataset(
                X_train_balanced,
                y_train_balanced,
                params={"verbosity": -1},
                free_raw_data=False,
            )
            lgb_val = lgb.Dataset(
                X_val_processed,
                y_val,
                reference=lgb_train,
                params={"verbosity": -1},
                free_raw_data=False,
            )
            t0_fit_time = time.time()
            model = lgb.train(
                params=lgb_params,
                train_set=lgb_train,
                valid_sets=[lgb_val],
                num_boost_round=n_estimators,
                feval=feval,
                callbacks=callbacks,
            )
            fit_time = time.time() - t0_fit_time

            # Si on utilise une fonc built-il pour la perte, le modèle renvoie déjà des probas,
            # Si on utilise les fonc perso, on a des logits et on doit appliquer sigmoid pour obtenir les probas,
            if self.loss == "binary":
                y_score_val = model.predict(X_val_processed)
            else:
                y_score_val = special.expit(model.predict(X_val_processed))
                # y_score_val = clip_sigmoid(model.predict(X_val_processed))
                # y_score_val = model.predict(X_val_processed)

            # Prédiction de la classe en fonction du seuil de probabilité
            y_pred_val = pd_pred_prob_to_binary(y_score_val, threshold=threshold_prob)

            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_val).ravel()

            # Mesures sur le jeu de validation
            dic_val_scores = {}
            dic_val_scores["auc"] = roc_auc_score(y_val, y_score_val)
            dic_val_scores["accuracy"] = accuracy_score(y_val, y_pred_val)
            dic_val_scores["recall"] = recall_score(y_val, y_pred_val)
            dic_val_scores["penalized_f1"] = penalize_f1(fp=fp, fn=fn, tp=tp)
            dic_val_scores["business_gain"] = penalize_business_gain(
                tn=tn, fp=fp, fn=fn, tp=tp
            )

            dic_val_scores["fit_time"] = fit_time
            dic_val_scores["tn"] = tn
            dic_val_scores["fn"] = fn
            dic_val_scores["tp"] = tp
            dic_val_scores["fp"] = fp

            del X_train
            del y_train
            del X_val
            del y_val
            del model
            gc.collect()

            # Dans optuna, il ne sera loggé par défaut que la métrique à optimiser, pour loguer toutes les métriques,
            # on personnalise pour enregistrer toutes les métriques
            # print("dic_val_scores", type(dic_val_scores))
            # print(dic_val_scores)
            mean_scores = {k: np.mean(v) for (k, v) in dic_val_scores.items()}
            trial.set_user_attr("mean_scores", mean_scores)

            # Fin d'un trial, on loggue dans mlflow
            mlflow.log_metrics(dic_val_scores)

            # On ne logue le dataset que dans le best_run
            # self.track_dataset()

        return dic_val_scores[self.metric]

    def hide_lgb_log(self):
        log_stream = io.StringIO()
        logging.basicConfig(stream=log_stream, level=logging.INFO, format="%(message)s")
        logger = logging.getLogger("lightgbm")
        logger.setLevel(logging.WARNING)  # Pour capturer les warnings
        # Configurer LightGBM pour utiliser ce logger
        lgb.register_logger(logger)
        return

    def run_lgb_trials(self, n_trials=10, verbose=False):

        # La verbosité dans les params ne fonctionnennt pas, donc pour ne pas afficher tout à l'écran, on redirige les logs vers une chaine de caractères.
        if not verbose:
            self.hide_lgb_log()

        if n_trials > 10:
            self.optuna_verbosity = optuna.logging.ERROR
        else:
            self.optuna_verbosity = optuna.logging.INFO
        optuna.logging.set_verbosity(self.optuna_verbosity)

        self._study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=1,
            gc_after_trial=True,
            # call_backs=pruning_callback,
        )

    def run(self, n_trials=10, verbose=False, lgb_n_threads=None):
        if lgb_n_threads:
            self.lgb_n_threads = lgb_n_threads

        if self.n_folds == 1:
            cv_str = " sans validation croisée"
        else:
            cv_str = f" en validation croisée {self.n_folds} folds"

        print(
            f"Optimisation de {n_trials} trials sur CPU ({self.lgb_n_threads} threads){cv_str}..."
        )
        t0 = time.time()

        self.run_lgb_trials(n_trials=n_trials, verbose=verbose)
        print("Durée de l'optimisation (hh:mm:ss) :", format_time(time.time() - t0))
        self.print_best_trial()
        print("Nombre total de trials élagués (=pruned) :", self.counts_pruned_trials())
        print()
        self.create_best_run()

    """def get_params_of_model(self):
        return {
            k: v
            for k, v in self.get_suggested_params().items()
            if k not in ["threshold_prob", "k_neighbors"] and not pd.isna(v)
        }"""

    def get_params_of_model(self):
        suggested_params = self.get_suggested_params()
        params_of_model = {
            k: v
            for k, v in suggested_params.items()
            if k not in ["threshold_prob", "k_neighbors", "balance"]
        }
        # On ajoute les paramètres du modèle que l'on a fixé :
        params_of_model["force_col_wise"] = True
        params_of_model["boosting_type"] = "gbdt"
        params_of_model["loss"] = self.loss
        return params_of_model


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
