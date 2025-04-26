import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.metrics import roc_auc_score as sk_roc_auc_score
from cuml.metrics import roc_auc_score as cu_roc_auc_score
from cuml.linear_model import LinearRegression as CuLinearRegression
from cuml.metrics import r2_score
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform, pdist
import os
import gc
import joblib
from imblearn.under_sampling import NearMiss
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split as sk_train_test_split
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.model_selection import StratifiedKFold as CuStratifiedKFold
import cudf
import cupy as cp

import cuml
from collections import defaultdict

import time


"""
# Pour l'install de nearmiss
# pip install -U imbalanced-learn
# Dépendances critiques : numpy >= 1.19.5, scipy >= 1.6.0 
https://imbalanced-learn.org/stable/install.html
-------
Python (>= 3.8)
NumPy (>= 1.17.3)
SciPy (>= 1.5.0)
Scikit-learn (>= 1.0.2)
https://pypi.org/project/imbalanced-learn/
"""

from src.p7_constantes import DATA_INTERIM, VAL_SEED
from src.p7_simple_kernel import get_memory_consumed

# , reduce_memory
from src.p7_preprocess import (
    check_no_nan,
    get_binary_features,
    get_null_variance,
    check_variances,
)
from src.p7_metric import business_gain_score
from src.p7_util import format_time


# Elimine récursivement les features qui ont le pire VIF tant qu'il existe des VIF supérieurs à un seuil.
# En validation croisée
# Le processus itératif peut s'arrêter avant d'atteindre le seuil si l'on spécifie une liste de features importantes.
# Dans ce cas, le processus s'arrête dès qu'une feature (le pire VIF) est considérée importante.
class VIFSelectorCV(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        pipe_prepare,
        n_splits=3,
        vif_threshold=10.0,
        important_features=[],
        verbose=True,
    ):
        self.pipe_prepare = pipe_prepare
        self.vif_threshold = vif_threshold
        self.important_features = important_features
        self.n_splits = n_splits
        self.verbose = verbose

    def fit(self, X, y=None, max_iter=30, epsilon=1e-6):
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        self.vif_means_ = {}
        self.selected_features_ = []
        self.removed_features_ = []

        t0 = time.time()
        if self.verbose:
            print(
                f"Elimination récursive de features avec un VIF > {self.vif_threshold:.0f}"
            )
            print(f"\tDevice : GPU")
            print(f"\tshape X  : {X.shape}")
            print(f"\tNombre de splits : {self.n_splits}\n")
            if max_iter > 0:
                print(f"Itérations ({max_iter} itérations maximum)...")
            else:
                print(f"Itérations...")

        # Sélection initiale : tous les prédicteurs
        current_features = [
            f for f in X.columns.tolist() if f not in ["TARGET", "SK_ID_CURR"]
        ]

        # en cuml le moddèle de régression linéaire exige que toutes les features soient typées en flotat64
        """not_float64 = [col for col in X.columns if X[col].dtype != "float64"]
        if not_float64:
            raise TypeError(
                f"Pour le VIF, toutes les colonnes de doivent être castées en float64"
            )"""

        iteration = 1

        # Tant que le pire VIF trouvé dans l'itération est supérieur au seuil fixé (vif_threshold)
        # On cherche le pire VIF et on supprime la feature correspondante
        # Toutefois on arrête le porcessus itératif si la pire feature trouvée fait partie des features importantes
        while True:
            vif_folds = {f: [] for f in current_features}
            # list_folds = cu_build_folds_list(5, X, y)
            folds = CuStratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=VAL_SEED,
            )

            for i_fold, (train_idx, val_idx) in enumerate(folds.split(X, y), start=1):
                X_train = X.iloc[train_idx][current_features].copy()
                y_train = y.iloc[train_idx]
                X_val = X.iloc[val_idx][current_features].copy()
                y_val = y.iloc[val_idx]

                X_train = self.pipe_prepare.fit_transform(X_train, y_train)
                X_val = self.pipe_prepare.transform(X_val)
                # On réunit les cudf verticalement
                X_fold = cudf.concat([X_train, X_val], axis=0, ignore_index=True)
                # On caste en float64 car c'est une exigence du modèle Régression linéaire de cuml
                X_fold = X_fold.astype(np.float64)

                # On libère la mem, sur GPU avec cuDF, cuML, les blocs mémoire sont réservés même après del,
                # ne pas oublier de vider le pool
                del X_train
                del X_val
                gc.collect()
                cp._default_memory_pool.free_all_blocks()

                # Dans certains folds, des features peuvent être supprimés à cause d'une variance nulle
                current_features_fold = X_fold.columns.tolist()

                # Calcul du VIF sur le fold en cours
                for col in current_features_fold:
                    # print("col", col)
                    col_target = X_fold[col]
                    X_others = X_fold.drop(columns=[col])

                    try:
                        # Inutile de copier X quand on fit, le dire explicitement évite un warning
                        model = CuLinearRegression(copy_X=False)
                        model.fit(X_others, col_target)
                        y_pred = model.predict(X_others)
                        r2 = r2_score(col_target, y_pred)
                        if (1.0 - r2) < epsilon:
                            vif = np.inf
                        else:
                            vif = 1.0 / (1.0 - r2)

                        vif_folds[col].append(float(vif))
                    except Exception as e:
                        if self.verbose:
                            print(f"VIF non calculé pour {col} (fold {i_fold}) : {e}")

                # On libère le fold
                del X_fold
                gc.collect()
                cp._default_memory_pool.free_all_blocks()

            # Moyenne des VIFs sur tous les folds, uniquement si on a au moins un VIF
            vif_mean = {
                col: np.mean(vif_values_for_col)
                for col, vif_values_for_col in vif_folds.items()
                if vif_values_for_col  # ne garde que si liste non vide
            }
            self.vif_means_ = vif_mean

            # Trouver le pire VIF. Si 2 VIF égaux (ex : np.inf), la première trouvée est la pire
            worst_feature = max(vif_mean, key=vif_mean.get)
            worst_vif = vif_mean[worst_feature]

            if self.verbose:
                print(
                    f"[Itération {iteration}] (Elapsed time : {format_time((time.time()-t0))}) Max VIF: {worst_feature} = {worst_vif:.2f}"
                )

            if worst_vif < self.vif_threshold:
                break

            #  si la pire feature trouvée est importante, on ne la supprime pas et on arrête le processus itératif
            if worst_feature in self.important_features:
                print(
                    f"Arrêt du processus car {worst_feature} est une feature importante"
                )
                break

            self.removed_features_.append(worst_feature)
            current_features.remove(worst_feature)

            if iteration == max_iter:
                print(
                    f"Arrêt du processus car le maximum d'itérations ({max_iter}) a été atteint"
                )
                break

            iteration += 1

        self.selected_features_ = current_features
        duration = format_time(time.time() - t0)
        if self.verbose:
            print(f"Durée du Fit (hh:mm:ss) : {duration}")
            print(f"{len(self.removed_features_)} features ont un VIF trop élevé :")
            print(self.removed_features_)
            print(f"{len(self.selected_features_)} features restantes")
        return self

    def transform(self, X):
        return X.drop([self.removed_features_], axis=1)

    def get_vif_means(self):
        return self.vif_means_


# Elimine les features uune par une récursivement à l'aide du VIF sans validation croisée
# Pour les datasets ne comportant pas de NaN uniquement.
class VIFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=10.0, verbose=True):
        # self.pipe_prepare = pipe_prepare
        self.vif_threshold = vif_threshold
        # self.epsilon = epsilon
        self.verbose = verbose

    def fit(self, X, y=None, max_iter=30, epsilon=1e-6):
        self.vif_ = {}
        self.selected_features_ = []
        self.removed_features_ = []

        t0 = time.time()
        if self.verbose:
            print(
                f"Elimination récursive de features avec un VIF > {self.vif_threshold:.0f}"
            )
            print(f"\tDevice : GPU")
            print(f"\tshape X  : {X.shape}")
            print(f"\tSans validation croisée\n")
            if max_iter > 0:
                print(f"Itérations ({max_iter} itérations maximum)...")
            else:
                print(f"Itérations...")

        current_features = [
            f for f in X.columns.tolist() if f not in ["TARGET", "SK_ID_CURR"]
        ]

        iteration = 1

        while True:
            X_current = X[current_features].copy()
            # X_prepared = self.pipe_prepare.fit_transform(X_current, y)
            # X_prepared = X_prepared.astype(np.float64)

            vif_scores = {}
            for col in X_current.columns:
                col_target = X_current[col]
                X_others = X_current.drop(columns=[col])

                try:
                    model = CuLinearRegression(copy_X=False)
                    model.fit(X_others, col_target)
                    y_pred = model.predict(X_others)
                    r2 = r2_score(col_target, y_pred)
                    vif = np.inf if (1.0 - r2) < epsilon else 1.0 / (1.0 - r2)
                    vif_scores[col] = float(vif)
                except Exception as e:
                    if self.verbose:
                        print(f"VIF non calculé pour {col} : {e}")

            self.vif_ = vif_scores

            worst_feature = max(vif_scores, key=vif_scores.get)
            worst_vif = vif_scores[worst_feature]

            if self.verbose:
                print(
                    f"[Itération {iteration}] (Elapsed time : {format_time((time.time()-t0))}) Max VIF: {worst_feature} = {worst_vif:.2f}"
                )

            if worst_vif < self.vif_threshold:
                break

            self.removed_features_.append(worst_feature)
            current_features.remove(worst_feature)

            if iteration == max_iter:
                print(
                    f"Arrêt du processus car le maximum d'itérations ({max_iter}) a été atteint"
                )
                break

            iteration += 1

            # Libération mémoire
            gc.collect()
            cp._default_memory_pool.free_all_blocks()

        self.selected_features_ = current_features
        duration = format_time(time.time() - t0)
        if self.verbose:
            print(f"Durée du Fit (hh:mm:ss) : {duration}")
            print(f"{len(self.removed_features_)} features ont un VIF trop élevé :")
            print(self.removed_features_)
            print(f"{len(self.selected_features_)} features restantes")
        return self

    def transform(self, X):
        return X.drop(columns=self.removed_features_, errors="ignore")

    def get_vif(self):
        return self.vif_


class CuVIFCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=10.0, verbose=True):
        self.vif_threshold = vif_threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        self.features_ = X.columns.tolist()
        df = X.copy()

        vif_dict = {}
        for col in df.columns:
            # La régression linéaire cuml n'accepte que des float, on caste si la colonne n'est pas float
            if df[col].dtype not in [np.float32, np.float64]:
                df[col] = df[col].astype(np.float32)
            y = df[col]
            X_others = df.drop(columns=[col])
            # Inutile de copier X quand on fit, (on travaille déjà sur une copie et on ne transforme pas X,
            # le dire explicitement évite un warning
            model = CuLinearRegression(copy_X=False)
            model.fit(X_others, y)
            y_pred = model.predict(X_others)
            r2 = r2_score(y, y_pred)
            vif = 1.0 / (1.0 - r2)
            vif_dict[col] = float(vif)

        self.vif_ = vif_dict
        self.feature_names_in_ = X.columns.tolist()
        return self

    def transform(self, X):
        check_is_fitted(self)
        return X.copy()  # Ne modifie pas les features (juste calcule les VIFs)

    def get_vif(self):
        check_is_fitted(self)
        return self.vif_


class CuCorrSelector(BaseEstimator, TransformerMixin):
    """
    Supprime les features redondantes c'est à dire trop corrélées 2 à 2
    """

    def __init__(
        self,
        exclude=[],
        method="pearson",
        rcaf=True,
        threshold_corr=0.9,
        threshold_dist=0.1,
        verbose=True,
    ):
        self.exclude = exclude
        self.method = method
        self.rcaf = rcaf
        self.threshold_corr = threshold_corr
        self.threshold_dist = threshold_dist
        self.verbose = verbose
        self._features_to_drop = []

    def fit(self, X, y):
        # Si exclude est vide on considère toutes les variables
        features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]
        if self.exclude:
            features = [f for f in features if f not in self.exclude]

        # Si des varaiables à étudier ont des variances nulles, on lève une erreur
        check_variances(X[features], raise_error=True)

        # On construit la matrice des corrélations:
        if self.rcaf:
            corr_matrix = build_rcaf_matrix(
                X[features],
                y=y,
                method=self.method,
            )
        else:
            corr_matrix = build_corr_matrix(
                X[features],
                corr_coef=self.method,
            )

        features_correlated_above_threshold = get_features_correlated_above(
            corr_matrix, self.threshold_corr, verbose=False
        )

        linkage_matrix = build_linkage_matrix(
            X,
            features_correlated_above_threshold,
            corr_coef=self.method,
            method="complete",
        )
        all_clusters_col, features_to_drop = cluster_features_from_linkage_matrix(
            X,
            linkage_matrix,
            features=features_correlated_above_threshold,
            threshold_pct=self.threshold_dist,
            verbose=self.verbose,
        )
        self._features_to_drop = features_to_drop

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = [
            f for f in self.feature_names_in_ if f not in self._features_to_drop
        ]
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_ = X[self.feature_names_out_].copy()
        if verbose:
            print(
                f"{len(self._features_to_drop)} features supprimées car redondantes :"
            )
            print(self._features_to_drop)
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class CuPermutSelector(BaseEstimator, TransformerMixin):
    """
    Supprime les features inutiles par permutation importances
    """

    def __init__(
        self,
        fitted_model,
        metric="auc",
        threshold_prob=0.5,
        threshold_importance=0,
        n_repeat=5,
        exclude=[],
        random_state=VAL_SEED,
        # verbose=True,
    ):
        self.fitted_model = fitted_model
        self.metric = metric
        self.threshod_prob = threshold_prob
        self.threshold_importance = threshold_importance
        self.n_repeat = n_repeat
        self.exclude = exclude
        # self.verbose = verbose
        self.random_state = random_state
        self.useless_features_ = []

    # Attention tout s'effectue sue le jeu de validation, mais X est nécessaire dans le tranform et il s'agit de X_train
    def fit(self, X, X_val, y_val, verbose=True):
        # Si exclude est vide on considère toutes les variables
        features = [f for f in X_val.columns if f not in ["SK_ID_CURR", "TARGET"]]
        if self.exclude:
            features = [f for f in features if f not in self.exclude]

        # Probabilité d'appartenir à la classe default pour le jeu de validation
        y_score_val = self.fitted_model.predict_proba(X_val)[1]

        # Mesure sur le jeu de validation avec toutes les colonnes intactes
        baseline_score = cuml.metrics.roc_auc_score(y_val, y_score_val)
        useless_features = []
        if verbose:
            print(f"Permutation importance ({self.n_repeat} répétitions)")
        # Pour chaque colonne on mélange la colonne, et on calcule la différence de score avec la baseline

        dic_importances = {col: [] for col in features}

        for col in features:
            for repeat in range(self.n_repeat):
                X_permuted = X_val.copy()

                # Pas de shuffle en cuda, Le fait de faire le sample sur 100% des données en ignorant l'index revient à effectuer un shuffle
                X_permuted[col] = (
                    X_permuted[col]
                    .sample(
                        frac=1,
                        replace=False,
                        ignore_index=True,
                        # On mélange différemment à chaque répétition
                        random_state=self.random_state + (repeat + 1) * 10,
                    )
                    .to_numpy()
                )

                permuted_score = cuml.metrics.roc_auc_score(
                    y_val, self.fitted_model.predict_proba(X_permuted)[1]
                )
                dic_importances[col].append(baseline_score - permuted_score)

        # On fait la moyenne des importances au travers des répétitions pour chaque colonne
        mean_importances = {k: np.mean(v) for k, v in dic_importances.items()}

        # On dresse la liste des features inutiles
        self.useless_features_ = [
            col
            for col in mean_importances.keys()
            if mean_importances[col] <= self.threshold_importance
        ]
        if verbose:
            print(f"{len(self.useless_features_)} features inutiles :")
            print(self.useless_features_)
        self.feature_names_in_ = X_val.columns.tolist()
        self.feature_names_out_ = [
            f for f in self.feature_names_in_ if f not in self.useless_features_
        ]
        return self

    # X est X_train, ne pas oublier d'enlever les features aussi sur X_val
    def transform(self, X, X_val, y_val=None, verbose=False):
        check_is_fitted(self)
        X_ = X[self.feature_names_out_].copy()
        # X_val_ = X_val[self.feature_names_out_].copy()
        if verbose:
            print(f"{len(self.useless_features_)} features supprimées car inutiles :")
            print(self.useless_features_)
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class CuPermutSelector2(BaseEstimator, TransformerMixin):
    """
    Supprime les features inutiles par permutation importances
    """

    def __init__(
        self,
        fitted_model,
        metric="auc",
        threshold_prob=0.5,
        threshold_importance=0,
        n_repeat=5,
        exclude=[],
        random_state=VAL_SEED,
        # verbose=True,
    ):
        self.fitted_model = fitted_model
        self.metric = metric
        self.threshod_prob = threshold_prob
        self.threshold_importance = threshold_importance
        self.n_repeat = n_repeat
        self.exclude = exclude
        # self.verbose = verbose
        self.random_state = random_state
        self.useless_features_ = []

    # Attention tout s'effectue sue le jeu de validation, mais X est nécessaire dans le tranform et il s'agit de X_train
    def fit(self, X_val, y_val, verbose=True):
        # Si exclude est vide on considère toutes les variables
        features = [f for f in X_val.columns if f not in ["SK_ID_CURR", "TARGET"]]
        if self.exclude:
            features = [f for f in features if f not in self.exclude]

        # Mesure sur le jeu de validation avec toutes les colonnes intactes
        # Probabilité d'appartenir à la classe default pour le jeu de validation
        y_score_val = self.fitted_model.predict_proba(X_val)[1]
        baseline_score = cu_roc_auc_score(y_val, y_score_val)
        useless_features = []
        if verbose:
            print(f"Permutation importance ({self.n_repeat} répétitions)")
        # Pour chaque colonne on mélange la colonne, et on calcule la différence de score avec la baseline

        dic_importances = {col: [] for col in features}

        for col in features:
            for repeat in range(self.n_repeat):
                X_permuted = X_val.copy()

                # Pas de shuffle en cuda, Le fait de faire le sample sur 100% des données en ignorant l'index revient à effectuer un shuffle
                X_permuted[col] = (
                    X_permuted[col]
                    .sample(
                        frac=1,
                        replace=False,
                        ignore_index=True,
                        # On mélange différemment à chaque répétition
                        random_state=self.random_state + (repeat + 1) * 10,
                    )
                    .to_numpy()
                )

                permuted_score = cu_roc_auc_score(
                    y_val, self.fitted_model.predict_proba(X_permuted)[1]
                )

                dic_importances[col].append(baseline_score - permuted_score)

        # On fait la moyenne des importances au travers des répétitions pour chaque colonne
        mean_importances = {k: np.mean(v) for k, v in dic_importances.items()}

        # On dresse la liste des features inutiles
        self.useless_features_ = [
            col
            for col in mean_importances.keys()
            if mean_importances[col] <= self.threshold_importance
        ]
        if verbose:
            print(f"{len(self.useless_features_)} features inutiles :")
            print(self.useless_features_)
        self.feature_names_in_ = X_val.columns.tolist()
        self.feature_names_out_ = [
            f for f in self.feature_names_in_ if f not in self.useless_features_
        ]
        return self

    # X est X_train, ne pas oublier d'enlever les features aussi sur X_val
    def transform(self, X, X_val, y_val=None, verbose=False):
        check_is_fitted(self)
        X_ = X[self.feature_names_out_].copy()
        # X_val_ = X_val[self.feature_names_out_].copy()
        if verbose:
            print(f"{len(self.useless_features_)} features supprimées car inutiles :")
            print(self.useless_features_)
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class PermutSelector_old(BaseEstimator, TransformerMixin):
    """
    Supprime les features inutiles par permutation importances
    """

    def __init__(
        self,
        fitted_model,
        metric="auc",
        device="cuda",
        threshold_prob=0.5,
        threshold_importance=0,
        n_repeat=5,
        exclude=[],
        random_state=VAL_SEED,
        # verbose=True,
    ):
        self.fitted_model = fitted_model
        self.metric = metric
        self.device = device
        self.threshod_prob = threshold_prob
        self.threshold_importance = threshold_importance
        self.n_repeat = n_repeat
        self.exclude = exclude
        # self.verbose = verbose
        self.random_state = random_state
        self.useless_features_ = []

    # Attention tout s'effectue sue le jeu de validation, mais X est nécessaire dans le tranform et il s'agit de X_train
    def fit(self, X, X_val, y_val, verbose=True):
        # Si exclude est vide on considère toutes les variables
        features = [f for f in X_val.columns if f not in ["SK_ID_CURR", "TARGET"]]
        if self.exclude:
            features = [f for f in features if f not in self.exclude]

        # Mesure sur le jeu de validation avec toutes les colonnes intactes
        if self.device == "cuda":
            # Probabilité d'appartenir à la classe default pour le jeu de validation
            y_score_val = self.fitted_model.predict_proba(X_val)[1]
            baseline_score = cu_roc_auc_score(y_val, y_score_val)
        else:
            y_score_val = self.fitted_model.predict_proba(
                X_val, num_iteration=self.fitted_model.best_iteration_
            )[:, 1]
            baseline_score = sk_roc_auc_score(y_val, y_score_val)
            print("baseline_score", baseline_score)
        useless_features = []
        if verbose:
            print(f"Permutation importance ({self.n_repeat} répétitions)")
        # Pour chaque colonne on mélange la colonne, et on calcule la différence de score avec la baseline

        dic_importances = {col: [] for col in features}

        for col in features:
            for repeat in range(self.n_repeat):
                X_permuted = X_val.copy()

                # Pas de shuffle en cuda, Le fait de faire le sample sur 100% des données en ignorant l'index revient à effectuer un shuffle
                X_permuted[col] = (
                    X_permuted[col]
                    .sample(
                        frac=1,
                        replace=False,
                        ignore_index=True,
                        # On mélange différemment à chaque répétition
                        random_state=self.random_state + (repeat + 1) * 10,
                    )
                    .to_numpy()
                )

                if self.device == "cuda":
                    permuted_score = cu_roc_auc_score(
                        y_val, self.fitted_model.predict_proba(X_permuted)[1]
                    )
                else:
                    permuted_score = sk_roc_auc_score(
                        y_val,
                        self.fitted_model.predict_proba(
                            X_permuted, num_iteration=self.fitted_model.best_iteration_
                        )[:, 1],
                    )

                dic_importances[col].append(baseline_score - permuted_score)

        # On fait la moyenne des importances au travers des répétitions pour chaque colonne
        mean_importances = {k: np.mean(v) for k, v in dic_importances.items()}

        # On dresse la liste des features inutiles
        self.useless_features_ = [
            col
            for col in mean_importances.keys()
            if mean_importances[col] <= self.threshold_importance
        ]
        if verbose:
            print(f"{len(self.useless_features_)} features inutiles :")
            print(self.useless_features_)
        self.feature_names_in_ = X_val.columns.tolist()
        self.feature_names_out_ = [
            f for f in self.feature_names_in_ if f not in self.useless_features_
        ]
        return self

    # X est X_train, ne pas oublier d'enlever les features aussi sur X_val
    def transform(self, X, X_val, y_val=None, verbose=False):
        check_is_fitted(self)
        X_ = X[self.feature_names_out_].copy()
        # X_val_ = X_val[self.feature_names_out_].copy()
        if verbose:
            print(f"{len(self.useless_features_)} features supprimées car inutiles :")
            print(self.useless_features_)
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class DataSelector:
    def __init__(
        self, num_prj="01", input_file="01_v0_simple_train.csv", random_state=42
    ) -> None:
        self.frac_test = 0.25
        self.random_state = random_state
        self.train = None
        self.test = None
        self.num_prj = num_prj
        self.input_dir = DATA_INTERIM
        self.input_file = input_file
        self.output_dir = DATA_INTERIM
        self.id = "SK_ID_CURR"
        self.target = "TARGET"
        self.not_predictors = ["TARGET", "SK_ID_CURR"]
        self.predictors = None
        # self.predictors = [
        #    f for f in self.train.columns.tolist() if f not in self.not_predictors
        # ]
        # On gardera ces features même si elles ont plus de 30 % de valeurs manquantes
        # car on sait qu'elles sont très importantes
        self.features_to_keep = [
            "EXT_SOURCE_1",
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "CREDIT_TO_ANNUITY_RATIO",  # annuity / credit dans simple kernel
            "APP_CREDIT_TO_ANNUITY_RATIO",  # credit / annuity dans full kernel
        ]
        self._binary_features = []
        self.resampling = ""

    # On garde le ratio de defaut
    def drop_rows_to_keep_feature(
        self,
        threshold=0.3,
        feature="EXT_SOURCE_1",
        keep_information=True,
        verbose=True,
        random_state=VAL_SEED,
    ):

        n_rows_before = self.train.shape[0]
        n_missing_before = self.train[feature].isna().sum(axis=0)
        ratio_before = n_missing_before / n_rows_before

        # S'il n'y a aucun besoin de supprimer des lignes (% de valeur manquantes en dessous du seuil toléré), on sort tout de suite
        if ratio_before <= 0.3:
            print(
                f"{feature} comporte {ratio_before:.0%} de valeurs manquantes (seuil toléré : {threshold:.0%}), aucune lignes à supprimer"
            )
            print("Forme du Train :", self.train.shape)
            return self.train

        # Si le % de valeurs manquantes de la feature est trop important,
        # On calcule le nombre de lignes à supprimer pour que la feature tombe à threshold % de valeurs manquantes
        ratio_after = 0.3
        # ratio_2 = n_missing_2 / (n_rows - n_rows_to_del)
        # ratio_2 * (n_rows - n_row_to_del) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1 + n_missing_2) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1) = n_missing_2 * (1 - ratio_2)
        # n_missing_2 = ratio_2 * (n_rows - n_missing_1) / (1 - ratio_2)
        n_missing_after = round(
            ratio_after * (n_rows_before - n_missing_before) / (1 - ratio_after)
        )
        n_rows_to_del = n_missing_before - n_missing_after

        # Afin de ne pas biaiser les données, on garde la proportion de défaut dans le jeu
        ratio_default = self.train["TARGET"].mean()
        if verbose:
            print(
                f"{feature}, comporte {n_missing_before:_} valeurs manquantes (ratio = {ratio_before:.0%})"
            )
            print(
                f"Proportion de Défauts avant suppression des lignes : {ratio_default:.1%}"
            )

        # Nombre de lignes à enlever parmi les défauts:
        n_rows_to_del_in_defaults = round(ratio_default * n_rows_to_del)
        n_rows_to_del_in_ok = round((1 - ratio_default) * n_rows_to_del)
        """print(
            f"On supprime {n_rows_to_del_in_defaults} lignes parmi les Défauts et {n_rows_to_del_in_ok} parmi les Ok"
        )"""

        if "SK_ID_CURR" in self.train.columns:
            self.train = self.train.set_index("SK_ID_CURR")

        # On sélectionne un subset dans lequel enlever les lignes qui est feature à NaN et target est OK
        subset_ok = self.train.loc[self.train["TARGET"] == 0, :]
        subset_to_drop_in = subset_ok.loc[subset_ok[feature].isna() == True, :]
        # print("subset_to_drop_in.shape", subset_to_drop_in.shape)

        if keep_information:
            # On calcule le nombre de manquantes par ligne pour enlever celles qui en ont le moins
            # On ne mélange pas avant car le train a déjà été mélangé lors du split et shuffle(df) n'existe pas dans cudf
            subset_to_drop_in["n_missing"] = subset_to_drop_in.isna().sum(axis=1)
            # On trie par nombre de missing les plus élevés en tête
            subset_to_drop_in = subset_to_drop_in.sort_values(
                by="n_missing", ascending=False
            )
            # On sélectionne le nombre de lignes parmi celles comportant le plus de valeurs manquantes
            idx_to_drop_ok = subset_to_drop_in.head(n_rows_to_del_in_ok).index

        else:
            # On tire au hazard (pas forcément le + de missings) des lignes à dropper parmi les OK
            # On trie par nombre de missing les plus élevés en tête
            idx_to_drop_ok = subset_to_drop_in.sample(
                n_rows_to_del_in_ok, random_state=random_state
            ).index

        # On sélectionne un subset dans lequel enlever les lignes qui est feature à NaN et target est Default
        subset_default = self.train.loc[self.train["TARGET"] == 1, :]
        subset_to_drop_in = subset_default.loc[
            subset_default[feature].isna() == True, :
        ]

        if keep_information:
            subset_to_drop_in["n_missing"] = subset_to_drop_in.isna().sum(axis=1)
            subset_to_drop_in = subset_to_drop_in.sort_values(
                by="n_missing", ascending=False
            )
            # On sélectionne le nombre de lignes parmi celles comportant le plus de valeurs manquantes
            idx_to_drop_default = subset_to_drop_in.head(
                n_rows_to_del_in_defaults
            ).index
        else:
            idx_to_drop_default = subset_to_drop_in.sample(
                n_rows_to_del_in_defaults, random_state=random_state
            ).index

        # On supprime ces lignes du train
        self.train = self.train.drop(idx_to_drop_ok)
        self.train = self.train.drop(idx_to_drop_default)
        self.train = self.train.reset_index()
        if verbose:
            print()
            print(
                f"Pourcentage de valeurs manquantes pour {feature} dans train après suppression de {len(idx_to_drop_ok) + len(idx_to_drop_default):_} lignes : {self.train[feature].isna().mean():.0%}"
            )
            print(
                f"Proportion de défaut après suppression : {self.train['TARGET'].mean():0.1%}"
            )
            print("Nouvelle forme du train", self.train.shape)
        return

    # Il semblerait qu'on introduise un biais si on enlève des lignes uniquement dans les sans défaut
    # Et que l'on supprime d'abord les lignes qui ont  le plus de NaN dans le jeu de données
    # Donc on va plutôt garder le ratio de défauts, et supprimer les lignes au hasard sans tenir compte de la perte d'information
    def drop_rows_to_keep_feature_old(
        self, threshold=0.3, feature="EXT_SOURCE_1", verbose=True
    ):

        n_rows_before = self.train.shape[0]
        n_missing_before = self.train[feature].isna().sum(axis=0)
        ratio_before = n_missing_before / n_rows_before

        # S'il n'y a aucun besoin de supprimer des lignes (% de valeur manquantes en dessous du seuil toléré), on sort tout de suite
        if ratio_before <= 0.3:
            print(
                f"{feature} comporte {ratio_before:.0%} de valeurs manquantes (seuil toléré : {threshold:.0%}), aucune lignes à supprimer"
            )
            print("Forme du Train :", self.train.shape)
            return self.train

        # Si le % de valeurs manquantes de la feature est trop important,
        # On calcule le nombre de lignes à supprimer pour que la feature tombe à threshold % de valeurs manquantes
        ratio_after = 0.3
        # ratio_2 = n_missing_2 / (n_rows - n_rows_to_del)
        # ratio_2 * (n_rows - n_row_to_del) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1 + n_missing_2) = n_missing_2
        # ratio_2 * (n_rows - n_missing_1) = n_missing_2 * (1 - ratio_2)
        # n_missing_2 = ratio_2 * (n_rows - n_missing_1) / (1 - ratio_2)
        n_missing_after = round(
            ratio_after * (n_rows_before - n_missing_before) / (1 - ratio_after)
        )
        n_rows_to_del = n_missing_before - n_missing_after
        if verbose:
            print(
                f"{feature}, comporte {n_missing_before:_} valeurs manquantes (ratio = {ratio_before:.0%})"
            )
            print(
                f"Pour tomber à {ratio_after:.0%} de valeurs manquantes, il nous faudrait descendre à {round(n_missing_after):_} valeurs manquantes"
            )

            print(f"Il nous faut donc supprimer {n_rows_to_del:_} lignes dans le train")

        # On calcule le pourcentage de valeur manquantes de la feature selon la target pour voir s'il y a une grosse différence
        subset_default = self.train.loc[self.train["TARGET"] == 1, [feature, "TARGET"]]
        ratio_default = subset_default[feature].isna().mean()
        subset_ok = self.train.loc[self.train["TARGET"] == 0, [feature, "TARGET"]]
        ratio_ok = subset_ok[feature].isna().mean()
        print(
            f"{feature} comporte {ratio_default:.0%} de valeurs manquantes pour la catégorie 'Défaut de remboursement' et {ratio_ok:.0%} pour la catégorie 'Remboursement OK'"
        )
        if abs(ratio_default - ratio_ok) < 0.1:
            print(
                "La différence entre les catégories n'étant pas importante, on supprime les lignes uniquement pour la catégorie 'Remboursement OK'"
            )
        else:
            print(
                "WARNING : Supprimer les lignes dans la catégorie 'Remboursement OK' uniquement peut introduire un biais"
            )

        if "SK_ID_CURR" in self.train.columns:
            self.train = self.train.set_index("SK_ID_CURR")

        # On sélectionne un subset dans lequel enlever les lignes qui est feature à NaN et target est OK
        subset_ok = self.train.loc[self.train["TARGET"] == 0, :]

        subset_to_drop_in = subset_ok.loc[subset_ok[feature].isna() == True, :]
        # print("subset_to_drop_in.shape", subset_to_drop_in.shape)

        # On calcule le nombre de manquantes par ligne pour enlever celles qui en ont le moins
        # On ne mélange pas avant car le train a déjà été mélangé lors du split et shuffle(df) n'existe pas dans cudf
        subset_to_drop_in["n_missing"] = subset_to_drop_in.isna().sum(axis=1)
        # On trie par nombre de missing les plus élevés en tête
        subset_to_drop_in = subset_to_drop_in.sort_values(
            by="n_missing", ascending=False
        )
        # On sélectionne le nombre de lignes parmi celles comportant le plus de valeurs manquantes
        idx_rows_to_drop = subset_to_drop_in.head(n_rows_to_del).index

        # On supprime ces lignes du train
        self.train = self.train.drop(idx_rows_to_drop)
        self.train = self.train.reset_index()
        if verbose:
            print("Nouvelle forme du train", self.train.shape)
            print(
                f"Pourcentage de valeurs manquantes pour {feature} dans train après suppression de {len(idx_rows_to_drop):_} lignes : {self.train[feature].isna().mean():.0%}"
            )
        return self.train

    def drop_too_missing_features(self, threshold=0.3, verbose=True):
        """Supprime les features comportant trop de valeurs manquantes,
        exceptées celles qui sont présisées dans l'attribut features_to_keep

        Args:
            threshold (float, optional): seuil toléré de valeurs manquantes. Defaults to 0.3.
            verbose (bool, optional): Verbosité. Defaults to True.

        Returns:
            cuDF: train débarrassé des features comportant trop de valeurs manquantes
        """
        # On identifie les features avec plus de threshold % de valeurs manquantes dans le train
        print("Suppression des features comportant trop de valeurs manquantes")
        too_missing_features = [
            f for f in self.train.columns if self.train[f].isna().mean() > threshold
        ]

        # On exclue de cette liste les features à garder même si elles comportent plus de valeurs manquantes que le seuil
        kept_although_too_missing = []
        for feature in self.features_to_keep:
            if feature in too_missing_features:
                too_missing_features.remove(feature)
                kept_although_too_missing.append(feature)

        shape_before = self.train.shape
        self.train.drop(too_missing_features, axis=1, inplace=True)
        # self.test.drop(too_missing_features, axis=1, inplace=True)

        if verbose:
            print("Forme du train avant suppression :", shape_before)
            print(
                f"{len(too_missing_features)} features supprimées du train comportant plus de {threshold:.0%} de valeurs manquantes"
            )
            if kept_although_too_missing:
                print(
                    f"{len(kept_although_too_missing)} features conservées bien que comportant plus de {threshold:.0%} de valeurs manquantes :"
                )
                print(kept_although_too_missing)
            print("Nouvelle forme de train :", self.train.shape)
        return

    def read_train_set(self, verbose=True):
        train = cudf.read_csv(os.path.join(self.input_dir, self.input_file))
        self.train = train
        if verbose:
            print("Informations sur le jeu de train :")
            print("type :", type(train))
            train.info()
        return

    # Inutile car on splitte dès le départ
    # Divise en train et test avec cuDF
    def split_cudf(self, verbose=True):
        """Divise en jeu de train et jeu de test (avec cuDF).
        Nécessite au moins 2 colonnes dans le jeu global sans valeurs manquantes.

        Args:
            verbose (bool, optional): Verbosité. Defaults to True.

        Returns:
            cuDF: Jeu de train
        """
        df = cudf.read_csv(os.path.join(self.input_dir, self.input_file)).set_index(
            "SK_ID_CURR"
        )
        # train_test_split en version cuda n'accepte pas les NaN,
        # nous allons donc l'utiliser avec des features sans NaN (au moins 2) pour récupérer les index.
        # Même ainsi, c'est beaucoup plus rapide qu'en passant par train_test_split de sklearn
        without_nan = [f for f in df.columns if df[f].isna().sum() == 0]
        if len(without_nan) < 2:
            print(
                "Pas assez de features sans NaN, faire une division manuelle ou utiliser sklearn"
            )
        else:
            X_train, X_test, _, _ = cu_train_test_split(
                df[without_nan[:2]],
                df["TARGET"],
                test_size=self.frac_test,
                stratify=df["TARGET"],
                random_state=self.random_state,
            )

            idx_train = X_train.index
            idx_test = X_test.index
            self.train = df.loc[idx_train].reset_index()
            self.test = df.loc[idx_test].reset_index()

            if verbose:
                print("Forme initiale du dataset :", df.shape)
                print(
                    f"train.shape : {self.train.shape}, test.shape : {self.test.shape}. (Fraction de test : {self.frac_test:.0%})"
                )
            del df
            gc.collect()
        return self.train

    # Elimine les features de variance nulle sur le train
    def drop_null_std(self, verbose=True):
        null_std_features = get_null_variance(self.train, verbose=verbose)
        if null_std_features:
            self.train = self.train.drop(null_std_features, axis=1)
            print(
                f"Nouvelle taille du jeu de Train : {self.train.shape}, {get_memory_consumed(self.train, verbose=False)} Mo"
            )
        return

    """def reduce_memory_usage(self, inplace=False):
        if not inplace:
            df = self.train.copy()
            reduce_memory(df)
            return df
        else:
            reduce_memory(self.train)
            reduce_memory(self.test)
            return"""

    def save_data(
        self,
        version="v1",
        suffix="",
        replace=False,
        train_rootname="train",
        # test_rootname="test",
    ):
        if suffix:
            train_name = f"{self.num_prj}_{version}_{train_rootname}_{suffix}.csv"
            # test_name = f"{self.num_prj}_{version}_{test_rootname}_{suffix}.csv"
        else:
            train_name = f"{self.num_prj}_{version}_{train_rootname}.csv"
            # test_name = f"{self.num_prj}_{version}_{test_rootname}.csv"

        # binary_features = self.get_binary_features()
        # binary_name = f"{self.num_prj}_{version}_binary_features.pkl"
        train_path = os.path.join(self.output_dir, train_name)
        # test_path = os.path.join(self.output_dir, test_name)
        # binary_path = os.path.join(self.output_dir, binary_name)

        save = True
        if not replace:
            if os.path.exists(train_path):
                print(f"Le fichier {train_path} existe déjà")
                print(
                    "Sauvegarde des données non effectuée. Modifiez la version ou forcer avec replace=True"
                )
                save = False
        if save:
            if "SK_ID_CURR" in self.train.columns:
                index = False
            else:
                index = True
            self.train.to_csv(train_path, index=index)
            print(f"Le fichier {train_path} sauvegardé. Forme {self.train.shape}")

            # On sauvegarde aussi la liste des features binaires
            # joblib.dump(binary_features, binary_path)
            # print(f"Liste des features binaires sauvegardée dans {binary_path}")

    # Devenue inutile ici car on ne s'embarasse plus du jeu de test
    def get_binary_features(self):
        # [TODO] lire le fichier de test
        all_data = cudf.concat([self.train, self.test])
        binary_features = get_binary_features(all_data)
        self._binary_features = binary_features
        return binary_features

    # [TODO]
    def useless_features(self, fitted_model):
        pass

    """
    I. 
    NearMiss version 1 :
    Sélectionne les échantillons de la classe majoritaire qui ont la plus petite moyenne des distances par rapport aux trois plus proches voisins de la classe minoritaire.
    => Cette version privilégie les échantillons de la classe majoritaire qui sont les plus proches en moyenne des échantillons de la classe minoritaire.
    
    Sélectionne les échantillons de la classe majoritaire pour chaque échantillon de la classe minoritaire
    qui a les plus petits m plus proches voisins de la classe majoritaire. 
    Cette version est différente des deux premières car elle se concentre sur chaque échantillon de la classe minoritaire individuellement.
    
    II. 
    NearMiss version 2:
    Sélectionne les échantillons de la classe majoritaire qui ont la plus petite distance maximale 
    par rapport aux trois plus proches voisins de la classe minoritaire. 
    =>  version privilégie les échantillons de la classe majoritaire qui sont proches des trois plus proches voisins de la classe minoritaire.
    
    Privilégie les échantillons de la classe majoritaire qui
    sont relativement proches de n'importe quel échantillon de la classe minoritaire. 
    Cela peut parfois être plus agressif que NearMiss-1.

    III. 
    NearMiss version 3:
    Sélectionne les échantillons de la classe majoritaire pour chaque échantillon de la classe minoritaire qui a les plus petits m plus proches voisins de la classe majoritaire. 
    Cette version est différente des deux premières car elle se concentre sur chaque échantillon de la classe minoritaire individuellement.
    
    Cette version est plus individualisée, 
    sélectionnant des échantillons de la classe majoritaire basés sur chaque échantillon de la classe minoritaire, 
    ce qui peut aider à capturer des nuances spécifiques de la classe minoritaire.

    """

    # [TODO] Décider enfin si on met l'id SK_ID_CURR en index ou pas, si on le garde ou non danws la target y
    def undersample_nearmiss(self, version=1, inplace=False, verbose=True):
        X = self.train.drop(self.target, axis=1)
        y = self.train[self.target]
        n_rows = X.shape[0]

        near_miss = NearMiss(version=version)
        X_resampled, y_resampled = near_miss.fit_resample(X, y)

        df_resampled = pd.DataFrame(
            X_resampled, columns=X.columns, index=self.train.index
        )
        df_resampled["TARGET"] = y_resampled
        if verbose:
            print(
                f"Taille actuelle de Train : {self.train.shape}, {get_memory_consumed(self.train)} Mo"
            )
            print(
                f"L'UnderSampling version {version} éliminerait {n_rows - self.train.shape[0]} lignes"
            )
            print(
                f"Aboutirait à Train de taille  : {df_resampled.shape}, {get_memory_consumed(df_resampled, verbose=False)} Mo"
            )
        if inplace:
            self.train = df_resampled
            if verbose:
                print("UnderSampling de Train effectué")
            return
        else:
            return X_resampled, y_resampled


# Réalise une élimination récursive de features sur GPU avec la librairie Rapids en s'inspirant de l'algorithme RFECV de sklearn
def cu_rfecv(
    X,
    y,
    estimator,
    pipe_prepare,
    cv=5,
    min_features=60,
    plot_curve=False,
    verbose=True,
    seed=42,
):
    features = list(X.columns)
    best_score = -np.inf
    best_features = features.copy()

    # Nombre de features restantes à chaque fois que l'on élimine une feature
    feature_counts = []
    # Score obtenu à chaque fois que l'on élimine une feature
    mean_scores = []

    # Tant qu'on n'a pas atteint le minimum de features à conserver,
    # on fit le modèle et on score le modèle en validation croisée,
    # puis on élimine la feature la moins importante
    while len(features) >= min_features:
        scores = []
        skf = CuStratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

        for train_idx, val_idx in skf.split(X[features], y):
            X_train, X_val = X[features].iloc[train_idx], X[features].iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            X_train = pipe_prepare.fit_transform(X_train)
            X_val = pipe_prepare.transform(X_val)

            model = estimator.fit(X_train[features], y_train)

            # y_prob = model.predict_proba(X_val[features])[:, 1]
            # y_prob = self.predict_proba(X)[1]
            # On ne gère pas de seuil de probabilité, il est donc à 0.5
            y_pred = model.predict(X_val).astype(int)

            # Mesures sur le jeu de validation
            # Order C applatit la matrice en ligne comme ravel() de numpy, mais
            # Attention, ne renvoie pas des scalaires mais des cupy ndarrays
            mat = cuml.metrics.confusion_matrix(y_val, y_pred)
            tn_array, fp_array, fn_array, tp_array = mat.ravel(order="C")
            # On extrait les scalaires des cupy ndarrays()
            tn = tn_array.item()
            fp = fp_array.item()
            fn = fn_array.item()
            tp = tp_array.item()

            score = business_gain_score(tp=tp, fn=fn, tn=tn, fp=fp)
            scores.append(score)

        mean_score = np.mean(scores)
        feature_counts.append(len(features))
        # On conserve le score obtenu dans la liste des scores obtenus à chaque itération
        mean_scores.append(mean_score)

        if verbose:
            print(f"{len(features)} features → Score = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_features = features.copy()

        # Si on atteint le minimum de features à conserver, on sort
        if len(features) <= min_features:
            break

        # L'importance des variables se mesure par l'attribut coeff_ pour les modèles linéaires (régression logistique)
        if hasattr(model, "coef_"):
            importances = cp.abs(model.coef_.to_cupy()).flatten()
        # Les modèles à base d'arbres fournissent l'attribut feature_importances (mais pas en cuml)
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            raise ValueError("Le modèle ne fournit pas d'importances de variables.")

        # On récupère la variable de plus faible importance et on la retire des features pour la prochaine itération
        min_idx = cp.argmin(importances).item()
        feature_to_remove = features[min_idx]
        features.pop(min_idx)

        if verbose:
            print(
                f"Suppression : '{feature_to_remove}' (importance={importances[min_idx]:.4f})"
            )

    return best_features, best_score, mean_scores


def get_features_correlated_above(corr_matrix, threshold_corr=0.0, verbose=True):
    correlations_above_threshold = []
    # On transforme en cupy array
    corr_array = corr_matrix.to_cupy()
    # On positionne un zéro sur la diagonale,
    # cela nous permettra de sélectionner les features qui ont une corr trop grande sans se sélectionner elles-mêmes
    cp.fill_diagonal(corr_array, 0.0)

    # On sélectionne les features qui ont une corrélation pair_wise trop élevée en valeur absolue par rapport au seuil
    corr_cudf = cudf.DataFrame(
        corr_array, columns=corr_matrix.columns, index=corr_matrix.columns
    )
    corr_ceiled = corr_cudf[
        (corr_cudf > threshold_corr) | (corr_cudf < -threshold_corr)
    ]
    corr_ceiled.dropna(axis=1, how="all", inplace=True)
    corr_ceiled.dropna(axis=0, how="all", inplace=True)
    correlations_above_threshold = corr_ceiled.columns.tolist()
    if verbose:
        print(
            f"{len(correlations_above_threshold)} features ont une corrélation > {threshold_corr:.2f} avec au moins une autre (en valeur absolue) :"
        )
        print(correlations_above_threshold)

    return correlations_above_threshold


# On ne calcule pas la matrice de corrélation dans la fonc et on ne vérifie pas
def sel_features_pairwise_correlation_old(
    df, method="spearman", ceil=0.95, sort_alpha=True, verbose=True
):
    check_no_nan(df)
    methods = ["pearson", "spearman", "kendall"]
    if method.lower() not in methods:
        raise ValueError(
            f"Les valeurs possibles de 'method' sont : {methods}, or method = {method}"
        )

    correlations = df.corr(method=method.lower())
    corr_array = correlations.to_numpy()
    # On positionne un zéro sur la diagonale,
    # cela nous permettra de sélectionner les features qui ont une corr trop grande sans se sélectionner elles-mêmes
    np.fill_diagonal(corr_array, 0.0)
    corr_cudf = cudf.DataFrame(
        corr_array, columns=correlations.columns, index=correlations.columns
    )
    corr_ceiled = corr_cudf[corr_cudf > ceil]
    corr_ceiled.dropna(axis=1, how="all", inplace=True)
    corr_ceiled.dropna(axis=0, how="all", inplace=True)
    too_correlated_features = corr_ceiled.columns.tolist()

    # On trie alphabétiquement le résultat si demandé
    if sort_alpha:
        too_correlated_features = sorted(too_correlated_features)
    if verbose:
        print(
            f"{len(too_correlated_features)} features ont un coefficient de corrélation {method}  > {ceil} avec au moins une autre :"
        )
        print(too_correlated_features)

    return too_correlated_features


# # ----------------Note Méthode Ward
# Même en utilisant les distances basées sur les corrélations, avec la méthode Ward (contrairement à Complete),
# On peut aboutir au final à des distances > 1
# En effet, Ward minimise la varaince intra-cluster et pour cela a une définition spécifique de la distance entre 2 clusters.
# Cette définition implique une transformation non linéaire de la distance initiale (comprise entre 0 et 1)
# Complete utilise la distance max entre les éléments de deux clusters, donc ça reste toujours entre 0 et 1
# ----------------Note Métrique
# Pour choisir entre métrique basée sur les corrélations ou clustering normal des features représentées dans l'espace des individus :
# Avantage corr : Directement liée à la similarité statistique entre les features (corrélations).
# Inconvénient corr : Nécessite une transformation des corrélations en distances, et les corrélations
#   peuvent ne pas capturer toutes les relations non linéaires entre les features.
# Avantage clust noraml : Peut capturer des similitudes complexes basées sur les valeurs individuelles des features, même si non linéaires.
# Inconvénient clust normal : Peut être plus sensible au bruit et aux valeurs aberrantes, moins intuitive que l'autre.
def cluster_features_old(
    data,
    to_process,
    corr_coef="spearman",
    threshold_pct=0.05,
    method="complete",
    figsize=None,
    to_mark=[],
):

    corr_coefficients = ["pearson", "spearman", "kendall", None]
    methods = ["ward", "complete", "average"]
    if corr_coef:
        corr_coef = corr_coef.lower()
    if corr_coef not in corr_coefficients:
        raise ValueError(
            f"Les valeurs possibles de 'corr_coef' sont : {corr_coefficients}, or corr_coef = {corr_coef}"
        )

    if method:
        method = method.lower()
    if method not in methods:
        raise ValueError(
            f"Les valeurs possibles de 'method' sont : {methods}, or method = {method}"
        )

    # La méthode n'accepte pas les NaN
    check_no_nan(data[to_process])

    # Pour pouvoir passer le seul en pourcentage sous la forme 0.01 ou 10%:
    if threshold_pct > 1:
        threshold_pct = threshold_pct / 100

    # On trie les features de to_process par ordre alphabétique car cela facilite la lecture et
    # les première features gardées parmi les clusters auront plus de chance d'être simples
    # car nom long = feature compliquée dans la plupart des cas.
    to_process = sorted(to_process)

    # Si corr_coeff n'est pas None, on effectue le regroupement par corrélations
    if corr_coef:
        # Métrique = 1-|corr| (2 features très corrélées sont très proches)
        corr = data[to_process].corr(method=corr_coef)
        # Correction pour avoir une matrice parfaitement symetrique
        corr = (corr + corr.T) / 2
        corr = corr.to_numpy()
        np.fill_diagonal(corr, 1.0)
        corr_dist = squareform(1 - abs(corr))

        # Réalisation du clustering hiérarchique
        if method == "complete":
            linkage_matrix = hierarchy.complete(corr_dist)
        elif method == "ward":
            linkage_matrix = hierarchy.ward(corr_dist)
        else:
            linkage_matrix = hierarchy.average(corr_dist)

    # Si corr_coeff est None, On effectue le clustering des features représentées dans l'espace des individus
    else:

        # Calcul de la matrice de distance
        X = data[to_process].to_numpy().T
        distance_matrix = pdist(X, metric="euclidean")

        # Calcul de la matrice de liaison
        linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method=method)

        """
        Si on souhaitait normaliser les distances
        # On normalise les distances dans la matrice de liaison pour l'exprimer entre 0 et 1 (en % de la distance max)
        max_distance = np.max(linkage_matrix[:, 2])
        linkage_matrix[:, 2] /= max_distance"""

    # print("distance max :", np.max(linkage_matrix[:, 2]))
    dist_max = np.max(linkage_matrix[:, 2])
    threshold_value = threshold_pct * dist_max

    # Récupération des clusters à partir de la matrice de linkage et du seuil de distance
    cluster_ids = hierarchy.fcluster(
        linkage_matrix, threshold_value, criterion="distance"
    )
    # On crée un dictionnaire pour les clusters
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # On récupère les clusters constitués de plus d'une variables (pour dropper les inutiles)
    clusters = [list(v) for v in cluster_id_to_feature_ids.values() if len(v) > 1]
    clusters_col = [list(data[to_process].columns[v]) for v in clusters]

    # On ne conserve que la première variable de chaque cluster, les autres sont retirées
    dropped_features = [data[to_process].columns[v[1:]] for v in clusters]
    dropped_features = [item for sublist in dropped_features for item in sublist]

    # On récupère tous les clusters (même si constitués d'une seule var) pour l'imprimer et le renvoyer
    all_clusters = [list(v) for v in cluster_id_to_feature_ids.values()]
    all_clusters_col = [list(data[to_process].columns[v]) for v in all_clusters]

    print(
        f"Au seuil de {threshold_pct:.0%} de la distance maximum (dist={threshold_value:.2f}): {len(all_clusters_col)} clusters :"
    )
    for cluster in all_clusters_col:
        print(f"\t{cluster}")
    print(f"{len(dropped_features)} variables à supprimer :", dropped_features)

    # ------------------ PLOT Dendrogramme

    # Si figsize est None, on calcule grossièrement la taille de la figure
    if not figsize:
        n_features = len(to_process)
        height_by_feature = 0.25
        title_margin = 0.5
        height = n_features * height_by_feature + title_margin
        width = 10
        figsize = (width, height)

    # On définit les titres
    if corr_coef:
        metric_name = f"1 - |corr| ({corr_coef.capitalize()})"
    else:
        metric_name = "distance euclidienne"
    title = f"Regroupement de features par clustering hiérarchique\nMéthode : Linkage={method.capitalize()}, métrique={metric_name}\n"

    fig, ax = plt.subplots(figsize=figsize)

    # plt.suptitle(title)
    ax.set_title(title, ha="left", x=0, fontsize=16)
    # ax.set_xlabel(f"Distance à chaque regroupement. Métrique : {metric_name}")
    ax.set_xlabel(f"Distance à chaque regroupement")

    dendro = hierarchy.dendrogram(
        linkage_matrix,
        orientation="right",
        ax=ax,
        labels=to_process,
    )
    # Ligne pour tracer le seuil
    ax.vlines(
        threshold_value,
        ymin=ax.get_ylim()[0],
        ymax=ax.get_ylim()[1],
        color="black",
        ls="--",
        lw=1,
        label="Seuil",
    )

    # On colorie en gris les labels des varaibles à dropper :
    if dropped_features:
        [
            t.set_color("grey")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in dropped_features
        ]

    # S'il ya des variables à distinguer,
    # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
    for feature in to_mark:
        not_in_to_process = [f for f in to_mark if f not in to_process]
        if not_in_to_process:
            print(
                f"Les features {not_in_to_process} ne seront pas 'marquées' car elles ne font pas partie de 'to_process'"
            )
        to_mark = [f for f in to_mark if f in to_process]

    if to_mark:
        [
            t.set_color("red")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark
        ]
        [
            t.set_fontweight("bold")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark and t.get_text() not in dropped_features
        ]

    # On positionne les ticks sur les axes et on diminue la taille de police des labels
    ax.tick_params(
        axis="both",
        left=True,
        bottom=True,
        labelsize=10,
        gridOn=False,
        grid_alpha=0.3,
        color="grey",
    )

    return all_clusters_col, dropped_features


def cluster_features(
    data,
    to_process,
    corr_coef="spearman",
    threshold_pct=0.05,
    method="complete",
):

    corr_coefficients = ["pearson", "spearman", "kendall", None]
    methods = ["ward", "complete", "average"]
    if corr_coef:
        corr_coef = corr_coef.lower()
    if corr_coef not in corr_coefficients:
        raise ValueError(
            f"Les valeurs possibles de 'corr_coef' sont : {corr_coefficients}, or corr_coef = {corr_coef}"
        )

    if method:
        method = method.lower()
    if method not in methods:
        raise ValueError(
            f"Les valeurs possibles de 'method' sont : {methods}, or method = {method}"
        )

    # La méthode n'accepte pas les NaN
    check_no_nan(data[to_process])

    # Pour pouvoir passer le seul en pourcentage sous la forme 0.01 ou 10%:
    if threshold_pct > 1:
        threshold_pct = threshold_pct / 100

    linkage_matrix = build_linkage_matrix(data, to_process, corr_coef, method)

    # print("distance max :", np.max(linkage_matrix[:, 2]))
    dist_max = np.max(linkage_matrix[:, 2])
    threshold_value = threshold_pct * dist_max

    # Récupération des clusters à partir de la matrice de linkage et du seuil de distance
    all_clusters_col, dropped_features = cluster_features_from_linkage_matrix(
        linkage_matrix,
        to_process,
        threshold_pct,
    )

    return all_clusters_col, dropped_features


# https://eprints.whiterose.ac.uk/134706/2/ELSEVI_3.pdf
def build_rcaf_matrix(X, y, method="pearson"):
    """
    Implémente le Robust Correlation Analysis Framework (RCAF) sans garder le signe du coefficient de corrélation.
    Atténue l'impact du déséquilibre des classes en effectuant de multiples sous-échantillonnages sans remise :
    https://eprints.whiterose.ac.uk/134706/2/ELSEVI_3.pdf

    Args:
        X (cudf.DataFrame): Données d'entrée sans NaN et sans feature de variance nulle.
        y (cudf.Series): Classes associées (0 ou 1).
        method (str): Méthode de calcul de la corrélation ('pearson' ou autre).

    Returns:
        cudf.DataFrame: Matrice des corrélations robustes en valeur absolue
    """
    # Séparer les classes
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    # Taille des classes
    n_class_0 = len(class_0)
    n_class_1 = len(class_1)

    # Identifier la classe majoritaire et minoritaire
    if n_class_0 > n_class_1:
        majority_class = class_0
        minority_class = class_1
    else:
        majority_class = class_1
        minority_class = class_0

    # Nombre de datasets équilibrés à créer
    M = ceil(len(majority_class) / len(minority_class))
    correlation_squares = []

    i = 0
    n_features = len(X.columns)
    matrix = cp.zeros((n_features, n_features))
    for _ in range(M):
        # Échantillonnage sans remplacement (= tirage sans remise)
        sampled_majority = majority_class.sample(n=len(minority_class), replace=False)
        balanced_dataset = cudf.concat([sampled_majority, minority_class])

        # Calculer la corrélation
        corr = balanced_dataset.corr(method=method)
        matrix += cp.square(corr.to_cupy())

    # Intégration des corrélations
    final_correlations = cudf.DataFrame(
        cp.sqrt(matrix / M), index=X.columns, columns=X.columns
    )
    return final_correlations


# inutilisée car la classe minoritaire a des variances nulles après séparation des classes
def build_weighted_corr_matrix(X, y, corr_coef="spearman", verbose=False):
    """Construit une matrice des corrélations pondérée par les classes. Lève une erreur si la matrice contient des NaN

    Args:
        X (cudf ou df): Données sans NaN et sans features de variance nulle
        corr_coef (str, optional): Type de corrélation à calculer. Defaults to "pearson".

    Raises:
        ValueError: Présence de NaN dans la matrice de corrélation

    Returns:
        cudf ou df: MAtrice des corrélations
    """
    if verbose:
        print(f"Calcul des corrélations {corr_coef.upper()} pondérées")
    corr_matrix = None

    class_0 = X[y == 0]
    print(class_0.head())
    print("Nan classe0", class_0.isna().sum().sum())
    class_1 = X[y == 1]
    print(class_1.head())
    print("Nan classe1", class_1.isna().sum().sum())

    # Vérification des variances dans chaque classe
    check_variances(class_0, raise_error=False)
    check_variances(class_1, raise_error=False)

    # On calcule les matrices de corrélations séparément pour chaque classe puis on pondère par le poids des classes
    correlation_class_0 = class_0.corr(method=corr_coef)
    correlation_class_1 = class_1.corr(method=corr_coef)
    weight_class_0 = len(class_0) / len(X)
    weight_class_1 = len(class_1) / len(X)
    corr_matrix = (correlation_class_0 * weight_class_0) + (
        correlation_class_1 * weight_class_1
    )

    # On vérifie sque la matrice ne contient pas de nan
    columns_with_nan = [
        f for f in corr_matrix.columns if corr_matrix[f].isna().sum() > 0
    ]
    # S'il reste encore des NaN après avoir supprimé les variances nulles, cela n'est pas normal
    if columns_with_nan:
        raise ValueError(
            f"La matrice des corrélations contient des NaN\n{columns_with_nan}"
        )
    return corr_matrix


def build_corr_matrix(X, corr_coef="spearman", verbose=True):
    """Construit une matrice des corrélations. Lève une erreur si la matrice contient des NaN

    Args:
        X (cudf ou df): Données sans NaN et sans features de variance nulle
        corr_coef (str, optional): Type de corrélation à calculer. Defaults to "pearson".

    Raises:
        ValueError: Présence de NaN dans la matrice de corrélation

    Returns:
        cudf ou df: MAtrice des corrélations
    """
    if verbose:
        print(f"Calcul des corrélations {corr_coef.upper()}")
    corr_matrix = None
    # On calcule la matrice des corrélations
    corr_matrix = X.corr(method=corr_coef)
    columns_with_nan = [
        f for f in corr_matrix.columns if corr_matrix[f].isna().sum() > 0
    ]
    # S'il reste encore des NaN après avoir supprimé les variances nulles, cela n'est pas normal
    if columns_with_nan:
        raise ValueError(
            f"La matrice des corrélations contient des NaN\n{columns_with_nan}"
        )
    return corr_matrix


def build_linkage_matrix(
    data,
    to_process,
    corr_coef="spearman",
    method="complete",
):

    # On trie les features de to_process par ordre alphabétique car cela facilite la lecture
    to_process = sorted(to_process)

    # Si corr_coeff n'est pas None, on effectue le regroupement par corrélations
    if corr_coef:
        # Métrique = 1-|corr| (2 features très corrélées sont très proches)
        corr = data[to_process].corr(method=corr_coef)
        # Correction pour avoir une matrice parfaitement symetrique
        corr = (corr + corr.T) / 2
        corr = corr.to_numpy()
        np.fill_diagonal(corr, 1.0)
        corr_dist = squareform(1 - abs(corr))

        # Réalisation du clustering hiérarchique
        if method == "complete":
            linkage_matrix = hierarchy.complete(corr_dist)
        elif method == "ward":
            linkage_matrix = hierarchy.ward(corr_dist)
        else:
            linkage_matrix = hierarchy.average(corr_dist)

    # Si corr_coeff est None, On effectue le clustering des features représentées dans l'espace des individus
    else:

        # Calcul de la matrice de distance
        X = data[to_process].to_numpy().T
        distance_matrix = pdist(X, metric="euclidean")

        # Calcul de la matrice de liaison
        linkage_matrix = hierarchy.linkage(squareform(distance_matrix), method=method)

        """
        Si on souhaitait normaliser les distances
        # On normalise les distances dans la matrice de liaison pour l'exprimer entre 0 et 1 (en % de la distance max)
        max_distance = np.max(linkage_matrix[:, 2])
        linkage_matrix[:, 2] /= max_distance"""

    return linkage_matrix


def cluster_features_from_linkage_matrix(
    X,
    linkage_matrix,
    features,
    threshold_pct=0.05,
    verbose=True,
):

    # print("distance max :", np.max(linkage_matrix[:, 2]))
    dist_max = np.max(linkage_matrix[:, 2])
    threshold_value = threshold_pct * dist_max

    # Récupération des clusters à partir de la matrice de linkage et du seuil de distance
    cluster_ids = hierarchy.fcluster(
        linkage_matrix, threshold_value, criterion="distance"
    )
    # On crée un dictionnaire pour les clusters
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # On récupère les clusters constitués de plus d'une variables (pour dropper les inutiles)
    clusters = [list(v) for v in cluster_id_to_feature_ids.values() if len(v) > 1]
    # clusters_col = [list(features[v]) for v in clusters]
    clusters_col = [list(X[features].columns[v]) for v in clusters]

    # On ne conserve que la première variable de chaque cluster, les autres sont retirées
    dropped_features = [X[features].columns[v[1:]] for v in clusters]
    dropped_features = [item for sublist in dropped_features for item in sublist]

    # On récupère tous les clusters (même si constitués d'une seule var) pour l'imprimer et le renvoyer
    all_clusters = [list(v) for v in cluster_id_to_feature_ids.values()]
    all_clusters_col = [list(X[features].columns[v]) for v in all_clusters]

    if verbose:
        """print(
            f"Au seuil de {threshold_pct:.0%} de la distance maximum (dist={threshold_value:.2f}): {len(all_clusters_col)} clusters :"
        )
        for cluster in all_clusters_col:
            print(f"\t{cluster}")"""
        print(f"{len(dropped_features)} variables redondantes :")
        print(dropped_features)

    return all_clusters_col, dropped_features


def plot_dendro(
    linkage_matrix,
    features,
    corr_coef="pearson",
    # method="complete",
    threshold_dist=0.04,
    dropped_features=[],
    title="Regroupement de features par clustering hiérarchique\n",
    subtitle="",
    # to_mark=[],
    verbose=True,
):
    # On calcule grossièrement la taille de la figure
    n_features = len(features)
    height_by_feature = 0.25
    title_margin = 2
    height = n_features * height_by_feature + title_margin
    width = 10
    figsize = (width, height)

    # On définit le sous_titre
    if not subtitle:
        metric_name = f"1 - |corr| ({corr_coef.capitalize()})"
        subtitle = f"Métrique={metric_name}, seuil distance : {threshold_dist:.0%}"

    fig, ax = plt.subplots(figsize=figsize)

    # ax.set_xlabel(f"Distance à chaque regroupement. Métrique : {metric_name}")
    ax.set_xlabel(f"Distance à chaque regroupement")

    dendro = hierarchy.dendrogram(
        linkage_matrix,
        orientation="right",
        ax=ax,
        labels=features,
    )
    # Ligne pour tracer le seuil
    if threshold_dist:
        ax.vlines(
            threshold_dist,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            color="black",
            ls="--",
            lw=1,
            label="Seuil",
        )

    # On colorie en gris les labels des varaibles à dropper :
    if dropped_features:
        [
            t.set_color("grey")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in dropped_features
        ]

    # S'il ya des variables à distinguer,
    # On récupère le texte des labels de l'axe des y et on colorie celui qui correspond
    """for feature in to_mark:
        not_in_to_process = [f for f in to_mark if f not in features]
        if not_in_to_process:
            print(
                f"Les features {not_in_to_process} ne seront pas 'marquées' car elles ne font pas partie de 'to_process'"
            )
        to_mark = [f for f in to_mark if f in features]

    if to_mark:
        [
            t.set_color("red")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark
        ]
        [
            t.set_fontweight("bold")
            for t in ax.yaxis.get_ticklabels()
            if t.get_text() in to_mark and t.get_text() not in dropped_features
        ]
"""
    # On positionne les ticks sur les axes et on diminue la taille de police des labels
    ax.tick_params(
        axis="both",
        left=True,
        bottom=True,
        labelsize=10,
        gridOn=False,
        grid_alpha=0.3,
        color="grey",
    )
    fig.suptitle(title)
    if subtitle:
        ax.set_title(subtitle, ha="left", x=0, fontsize=ax.xaxis.label.get_fontsize())
    plt.tight_layout()

    if not verbose:
        # On n'affiche pas la figure
        plt.close(fig)
    return fig


# df_importances contient en colonnes les features et en lignes les folds
def plot_permutation_importance(
    importances, title="Permutation importance sur 5 folds\n", figsize=(10, 10)
):
    top = 10
    importances_mean = importances.mean()
    most_important_features = importances_mean.sort_values(ascending=True).tail(top)
    less_important_features = importances_mean.sort_values(ascending=True).head(top)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize)

    fig.suptitle(title, fontsize=14, ha="left")

    # ax1 = importances_df[most_important_features.index].plot.box(vert=False, whis=10)
    importances[most_important_features.index].plot.box(vert=False, whis=10, ax=ax1)
    ax1.set_title("Les plus importantes en moyenne", ha="left", x=0)
    ax1.axvline(x=0, color="k", linestyle="--")
    ax1.set_xlabel("Perte en AUC score")
    _, xlim_max = ax1.get_xlim()

    importances[less_important_features.index].plot.box(vert=False, whis=10, ax=ax2)
    ax2.set_title("Les moins importantes en moyenne", ha="left", x=0)
    ax2.axvline(x=0, color="k", linestyle="--")
    ax2.set_xlabel("Perte en AUC score")
    xlim_min, _ = ax2.get_xlim()

    """ax1.set_xlim(xlim_min, xlim_max)
    ax2.set_xlim(xlim_min, xlim_max)"""
    fig.tight_layout()
    return fig


def plot_top_correlations(
    corr_matrix,
    max_n_features=10,
    title="Matrice des corrélations des features les plus corrélées entre elles",
    subtitle="",
    figsize=(8, 8.3),
    symetric_cmap=False,
    verbose=True,
):
    features = corr_matrix.columns.tolist()
    if isinstance(corr_matrix, cudf.DataFrame):
        pd_corr_matrix = corr_matrix.to_pandas()
    else:
        pd_corr_matrix = corr_matrix
    if len(features) > max_n_features:
        # On place zéro sur la diagonale afin de ne pas sélectionner ces corrélations
        np.fill_diagonal(pd_corr_matrix.to_numpy(), 0.0)
        # Get the top absolute correlations
        top_correlations = pd_corr_matrix.abs().unstack().sort_values(ascending=False)
        top_correlations = top_correlations[:max_n_features]

        # Get the corresponding columns
        top_correlation_cols = set()
        for col1, col2 in top_correlations.index:
            top_correlation_cols.add(col1)
            top_correlation_cols.add(col2)

        # On replace 1 sur la diagonale
        np.fill_diagonal(pd_corr_matrix.to_numpy(), 1.0)

        # Plot correlation matrix using Seaborn
        fig, ax = plt.subplots(figsize=figsize)

        # Si les corrélations ne sont pas en valeur absolue
        if symetric_cmap:
            cmap = symmetrical_colormap(cmap_settings=("Blues", None), new_name=None)
            vmin = -1
        # Si les corrélations sont en valeurs absolues
        else:
            cmap = "Blues"
            vmin = 0

        sns.heatmap(
            pd_corr_matrix.loc[list(top_correlation_cols), list(top_correlation_cols)],
            vmin=vmin,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            cmap=cmap,
            linewidths=0.1,
            yticklabels=True,
            # Colorbar : rétrécie en hauteur et en largeur
            cbar_kws={"format": "%.2f", "fraction": 0.035},
            annot_kws={"fontsize": 10},
            ax=ax,
        )
        fig.suptitle(title)
        if subtitle:
            ax.set_title(
                subtitle,
                ha="left",
                x=0,
                fontsize=ax.xaxis.label.get_fontsize(),
            )
        if not verbose:
            plt.close(fig)
        return fig


def symmetrical_colormap(cmap_settings=("Blues", None), new_name=None):
    """Crée une colormap symétrique (pour les matrices de corrélation) en concaténant une cmap simple qui va d'une couleur vers le blanc. (ex : "Blues")
    avec son inverse

    Args:
        cmap_settings (_type_): _description_
        new_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Crée la colormap simple qui va de la couleur vers le blanc
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # Finesse de définition de la cmap
    n = 128

    # On récupère la liste des couleurs de la cmap simple qui feront la partie droite de la nouvelle cmpa
    colors_r = cmap(np.linspace(0, 1, n))
    # On renverse l'ordre des couleurs pour faire la partie gauche de la nouvelle cmpa
    colors_l = colors_r[::-1]

    # On combine partie gauche et droite pour faire la nouvelle cmap
    colors = np.vstack((colors_l, colors_r))
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap
