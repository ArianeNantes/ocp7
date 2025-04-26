import numpy as np
import pandas as pd

import cudf
import cuml
import cupy as cp
import sklearn
from cuml.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from cuml.neighbors import NearestNeighbors
from cuml.linear_model import LinearRegression as CuLinearRegression
from cuml.metrics import r2_score
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.impute import SimpleImputer as skSimpleImputer
from sklearn.preprocessing import RobustScaler as skRobustScaler
from sklearn.preprocessing import MinMaxScaler as skMinMaxScaler
from sklearn.feature_selection import VarianceThreshold as skVarianceThreshold
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, SMOTENC
import gc
import os
import warnings

from copy import deepcopy

# from sklearn.cluster import AgglomerativeClustering
# from collections import defaultdict

from src.p7_constantes import VAL_SEED, DATA_INTERIM

warnings.filterwarnings(
    "ignore",
    message="Unused keyword parameter: n_jobs during cuML estimator initialization",
)


class WithColNames(BaseEstimator, TransformerMixin):
    """
    Wrapper qui contient un estimateur cuml comme StandardScaler mais qui permet de garder les noms des colonnes
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist()
        self.estimator.fit(X)
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_columns = X.columns
        if self.estimator.copy:
            X_ = X.copy()
        else:
            X_ = X
        X_ = self.estimator.transform(X_)
        X_.columns = X_columns
        self.feature_names_out_ = X_.columns.tolist()
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


# Imputation en fonction du type : Mode pour les Bool, médiane pour les floats, partie entière de la médiane pour les int.
class ImputerByDtype(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.dic_median = {}
        self.dic_mode = {}

    def fit(self, X, y=None):

        self.bool_features_ = []
        self.other_features_ = []
        self.bad_type_features_ = []
        for f in X.columns:
            dtype = X[f].dtype
            dtype_str = str(dtype)

            # Gestion des booléens (imputation par le mode)
            if "bool" in dtype_str:
                self.bool_features_.append(f)
                mode_series = X[f].mode()
                if len(mode_series) > 0:
                    self.dic_mode[f] = mode_series[0]
                else:
                    self.dic_mode[f] = False  # ou un choix par défaut

            # Autres types (imputation par médiane en respectant le type)
            else:
                self.other_features_.append(f)
                median_val = X[f].dropna().median()

                if dtype == np.float16 or dtype == "float16":
                    self.dic_median[f] = np.float16(median_val)
                elif dtype == np.float32 or dtype == "float32":
                    self.dic_median[f] = np.float32(median_val)
                elif dtype == np.float64 or dtype == "float64":
                    self.dic_median[f] = np.float64(median_val)
                elif "int" in dtype_str:
                    # On prend la partie entière de la médiane et on la caste au type python correspondant
                    self.dic_median[f] = X[f].dtype.type(int(median_val))
                else:
                    self.bad_type_features_.append(f)
        if self.verbose and self.bad_type_features_:
            print(
                f"WARNING : {len(self.bad_type_features_)} features ont un type non géré par l'imputer"
            )
        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        X_copy = X.copy()
        for f in self.bool_features_:
            X_copy[f] = X_copy[f].fillna(self.dic_mode[f])
        for f in self.other_features_:
            X_copy[f] = X_copy[f].fillna(self.dic_median[f])

        if self.verbose and X_copy.isna().sum().sum() > 0:
            print("WARNING : Il reste des NaN")
        return X_copy

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class Imputer(BaseEstimator, TransformerMixin):
    """
    Impute sur place les features binaires avec le mode et les autres avec la médiane
    """

    def __init__(
        self,
        to_impute=[],
        binary_features=[],
        ordinal_features=["AGE_RANGE"],
        cast_bool=True,
        verbose=True,
    ):
        # print('\n***** init()')
        # Liste des noms de colonnes à imputer
        self.to_impute = to_impute
        self.binary_features = binary_features
        self.ordinal_features = ordinal_features
        self.cast_bool = cast_bool
        self.verbose = verbose

    def fit(self, X, y=None):
        # Si to_impute n'est pas précisé, on considère toutes les features
        if not self.to_impute:
            self.to_impute_ = [
                f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]
            ]
        else:
            self.to_impute_ = [f for f in self.to_impute if f in X.columns]

        # Si binary_features n'est pas précisé, on les identifie (uniquement celles à imputer).
        # Attention en cas de validation croisée, des features peuvent se retrouver binaires dans le train
        # (si valeurs 0 ou 1) et non binaires dans le jeu de validation
        if not self.binary_features:
            self.binary_features_ = get_binary_features(X)
        else:
            self.binary_features_ = [f for f in self.binary_features if f in X.columns]

        bool_to_impute = [f for f in self.binary_features_ if f in self.to_impute_]
        ordinal_to_impute = [
            f
            for f in self.ordinal_features
            if f in self.to_impute_ and f not in bool_to_impute
        ]
        self.to_impute_with_mode_ = bool_to_impute + ordinal_to_impute
        self.to_impute_with_median_ = [
            f for f in self.to_impute_ if f not in self.to_impute_with_mode_
        ]

        # On construit un dictionnaire pour les features qui seront à imputer avec le mode
        # Les clefs sont les noms des features et les valeurs sont les modes.
        # cudf renvoie une valeur négative si plusieurs valeurs sont possibles. Ici on renvoie la plus petite valeur (false) comme Pandas
        self.dic_impute_mode_ = {
            f: X[f].dropna().mode()[0] for f in self.to_impute_with_mode_
        }

        # On construit un dictionnaire pour les features qui seront à imputer avec la médiane
        # self.df_median = X[self.to_impute_with_median].median(axis=0)
        self.dic_impute_median_ = {
            f: X[f].dropna().median() for f in self.to_impute_with_median_
        }

        # L'underscore à la fin d'une variable dans la méthode fit permettra à la fonction check_is_fitted
        # de vérifier si l'estimateur a été fitté (c'est pourquoi on ne positionne pas l'attribut dans l'init)
        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X, y=None, verbose=False):
        # Si l'estimateur n'a pas été fitté, on sort avec un message d'erreur correct
        check_is_fitted(self)

        # Si l'estimateur a été fitté, on continue
        # La transformation n'est pas effectuée sur place
        X_ = X.copy()

        # [TODO] A voir si on réduit la transformation aux features qui contiennent des NaN
        # self.features_with_nan = [f for f in X.columns if X[f].isna().sum() > 0]

        # On impute avec les modes calculés lors du fit pour éviter la fuite de données
        for k in self.dic_impute_mode_.keys():
            X_.loc[X_[k].isna(), k] = self.dic_impute_mode_[k]

        # On impute avec les médianes calculées lors du fit pour éviter la fuite de données
        for k in self.dic_impute_median_.keys():
            X_.loc[X_[k].isna(), k] = self.dic_impute_median_[k]

        # Si demandé on caste les bools qui font partie de l'imputation
        if self.cast_bool:
            cast_to_bool = [f for f in self.binary_features_ if f in self.to_impute_]
            X_[cast_to_bool] = X_[cast_to_bool].astype("bool")
        self.feature_names_out_ = X_.columns.tolist()

        # Il ne devrait plus rester de valeurs manquantes parmi les features à imputer
        if X_[self.to_impute_].isna().sum().sum() > 0:
            print(
                f"WARNING : il reste des valeurs manquantes parmi les features à imputer"
            )
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class VarianceSelector(BaseEstimator, TransformerMixin):
    """
    Supprime les colonnes dont la variance est <= threshold
    """

    def __init__(self, threshold=0, copy=True, verbose=False):
        self.threshold = threshold
        self.copy = copy
        self.verbose = verbose

    def fit(self, X, y=None):
        self.to_drop_ = []
        dtype_list = X.dtypes.value_counts().index.to_numpy().tolist()
        # Pour chaque type de colonnes, on calcule les variances et on sélectionne les features de variance trop faibles
        for dt in dtype_list:
            dt_variances = X.select_dtypes(dt).var(
                axis=0, skipna=True, numeric_only=False
            )
            self.to_drop_ = (
                self.to_drop_
                + dt_variances[dt_variances <= self.threshold].index.to_numpy().tolist()
            )
        if self.verbose:
            print("Features de variances nulle", self.to_drop_)

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = [
            f for f in self.feature_names_in_ if f not in self.to_drop_
        ]
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        if self.copy:
            X_ = X.copy()
        else:
            X_ = X

        if self.to_drop_:
            # On supprime les colonnes de variances trop faibles
            X_.drop(self.to_drop_, axis=1, inplace=True)
        else:
            X_ = X
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class CuStandardScaler(BaseEstimator, TransformerMixin):
    """
    Scale les variables un Standard Scaler
    """

    def __init__(
        self, to_scale=[], features_binary=[], exclude_binary=True, verbose=True
    ):
        self.to_scale = to_scale
        self.features_binary = features_binary
        self.exclude_binary = exclude_binary
        self.verbose = verbose

    def fit(self, X, y=None):
        # Si to_scale n'a pas été spécifié, on considère d'abord toutes les variables
        if not self.to_scale:
            features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]
        else:
            features = [f for f in self.to_scale if f in X.columns]

        # Si on doit exclure les variables binaires,
        # si elles ne sont pas précisées on les identifie afin de ne pas les mettre à l'échelle
        if self.exclude_binary:
            if not self.features_binary:
                features_binary = get_binary_features(X[features])
            else:
                features_binary = [f for f in self.features_binary if f in X.columns]
            self.to_scale_ = [f for f in features if f not in features_binary]
        else:
            self.to_scale_ = features

        # Si des varaiables à mettre à l'échelle ont des variances nulles, on lève une erreur
        check_variances(X[self.to_scale_], raise_error=True)

        self.scaler_ = StandardScaler(with_mean=True, with_std=True)

        # self.scaler_.fit(X[self.to_scale_].to_numpy())
        self.scaler_.fit(X[self.to_scale_].values)
        self.mean_ = self.scaler_.mean_
        self.scale_ = self.scaler_.scale_
        # n_samples_seen_ provoque une erreur d'index si autre type que cparray
        self.n_samples_seen_ = self.scaler_.n_samples_seen_

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_ = X.copy()
        scaled_features = cudf.DataFrame(
            self.scaler_.transform(X_[self.to_scale_].values),
            columns=self.to_scale_,
            index=X.index,
        )
        for feature in self.to_scale_:
            X_[feature] = scaled_features[feature]
        self.feature_names_out_ = X_.columns.tolist()
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class CuRobustScaler(BaseEstimator, TransformerMixin):
    """
    Scale les variables non booléennes avec un RobustScaler
    """

    def __init__(
        self,
        to_scale=[],
        features_binary=[],
        exclude_binary=True,
        quantile_range=(25, 75),
        verbose=True,
    ):
        self.to_scale = to_scale
        self.features_binary = features_binary
        self.exclude_binary = exclude_binary
        self.quantile_range = quantile_range
        self.verbose = verbose

    def fit(self, X, y=None):
        # Si to_scale n'a pas été spécifié, on considère d'abord toutes les variables
        if not self.to_scale:
            features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]
        else:
            features = [f for f in self.to_scale if f in X.columns]

        # Si on doit exclure les variables binaires, on les identifie afin de ne pas les mettre à l'échelle
        if self.exclude_binary:
            if not self.features_binary:
                features_binary = get_binary_features(X[features])
            else:
                features_binary = [f for f in self.features_binary if f in X.columns]
            self.to_scale_ = [f for f in features if f not in features_binary]
        else:
            self.to_scale_ = features

        self.robust_scaler_ = RobustScaler(
            with_centering=True, with_scaling=True, quantile_range=self.quantile_range
        )

        # Si des varaiables à mettre à l'échelle ont des variances nulles, on lève une erreur
        check_variances(X[self.to_scale_], raise_error=True)

        self.robust_scaler_.fit(X[self.to_scale_].values)
        self.center_ = self.robust_scaler_.center_
        self.scale_ = self.robust_scaler_.scale_

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_ = X.copy()
        scaled_features = cudf.DataFrame(
            self.robust_scaler_.transform(X_[self.to_scale_].values),
            columns=self.to_scale_,
            index=X.index,
        )
        for feature in self.to_scale_:
            X_[feature] = scaled_features[feature]
        self.feature_names_out_ = X_.columns.tolist()
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class CuMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scale les variables non booléennes avec un MinMaxScaler
    """

    def __init__(
        self,
        to_scale=[],
        feature_range=(0, 1),
        features_binary=[],
        exclude_binary=True,
        verbose=True,
    ):
        self.to_scale = to_scale
        self.feature_range = feature_range
        self.features_binary = features_binary
        self.exclude_binary = exclude_binary
        self.verbose = verbose

    def fit(self, X, y=None):
        # Si to_scale n'a pas été spécifié, on considère d'abord toutes les variables
        if not self.to_scale:
            features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]
        else:
            features = [f for f in self.to_scale if f in X.columns]

        # Si on doit exclure les variables binaires, on les identifie afin de ne pas les mettre à l'échelle
        if self.exclude_binary:
            if not self.features_binary:
                features_binary = get_binary_features(X[features])
            else:
                features_binary = [f for f in self.features_binary if f in X.columns]
            self.to_scale_ = [f for f in features if f not in features_binary]
        else:
            self.to_scale_ = features

        # Si des varaiables à mettre à l'échelle ont des variances nulles, on lève une erreur
        check_variances(X[self.to_scale_], raise_error=True)

        self.scaler_ = MinMaxScaler(feature_range=self.feature_range)

        self.scaler_.fit(X[self.to_scale_].values)
        # Documentation RAPIDS : https://docs.rapids.ai/api/cuml/stable/api/#cuml.preprocessing.MinMaxScaler
        self.scale_ = self.scaler_.scale_
        self.min_ = self.scaler_.min_
        self.data_range_ = self.scaler_.data_range_
        self.n_samples_seen_ = self.scaler_.n_samples_seen_

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = X.columns.tolist()
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_ = X.copy()
        scaled_features = cudf.DataFrame(
            self.scaler_.transform(X_[self.to_scale_].values),
            columns=self.to_scale_,
            index=X.index,
        )
        for feature in self.to_scale_:
            X_[feature] = scaled_features[feature]
        self.feature_names_out_ = X_.columns.tolist()
        return X_

    def get_feature_names_in(self):
        check_is_fitted(self)
        return self.feature_names_in_

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.feature_names_out_


class Preprocessor:
    def __init__(self, device="cpu", model_name="logreg"):
        self.device = device
        self.model_name = model_name
        self.numeric_transformer = None
        self.binary_transformer = None

    def build(self):
        if self.device == "cpu":
            if self.model_name == "logreg":
                # Pipeline pour les colonnes numériques
                self.numeric_transformer = skPipeline(
                    steps=[
                        (
                            "imputer",
                            skSimpleImputer(strategy="median"),
                        ),
                        ("variance_threshold", skVarianceThreshold(threshold=0.0)),
                        ("robust_scaler", skRobustScaler()),
                        ("minmax_scaler", skMinMaxScaler()),
                    ]
                )
                # Pipeline pour les colonnes binaires
                self.binary_transformer = skPipeline(
                    steps=[("imputer", skSimpleImputer(strategy="most_frequent"))]
                )
            return self.binary_transformer, self.numeric_transformer


def check_no_nan(df):
    features_with_nan = [f for f in df.columns if df[f].isna().sum() > 0]
    if features_with_nan:
        raise ValueError(
            f"Eliminez d'abord les NaN, {len(features_with_nan)} features ont des valeurs manquantes :\n{features_with_nan}"
        )


def check_variances(df, raise_error=True, verbose=True):
    null_var = []
    dtype_list = df.dtypes.value_counts().index.to_numpy().tolist()
    # Pour chaque type de colonnes, on calcule les variances et on sélectionne les features de variance trop faibles
    for dt in dtype_list:
        dt_variances = df.select_dtypes(dt).var(axis=0, skipna=True, numeric_only=False)
        null_var = null_var + dt_variances[dt_variances <= 0].index.to_numpy().tolist()
    if null_var:
        if raise_error:
            raise ValueError(
                f"Eliminez d'abord les variables de variance nulle, {len(null_var)} features :\n{null_var}"
            )
        else:
            if verbose:
                print(
                    f"WARNING : {len(null_var)} variables de variance nulle :\n{null_var}"
                )
    return null_var
    """else:
        return []"""


def get_null_variance(df, verbose=True):
    selector = VarianceSelector(verbose=False)
    selector.fit(df)
    if verbose:
        print(f"{len(selector.to_drop_)} features de variance nulle")
        if selector.to_drop_:
            print(selector.to_drop_)
    return selector.to_drop_


def get_binary_features(df):
    features = [f for f in df.columns if f not in ["TARGET", "SK_ID_CURR"]]
    features_bool = (
        df[features].select_dtypes(include=["bool", "bool_"]).columns.tolist()
    )
    features_not_bool = [f for f in features if f not in features_bool]

    if isinstance(df, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        features_bin_not_bool = [
            f
            for f in features_not_bool
            if df[f].dropna().nunique() <= 2
            and all(value in [0, 1] for value in df[f].dropna().unique().to_numpy())
        ]
    else:
        features_bin_not_bool = [
            f
            for f in features_not_bool
            if df[f].dropna().nunique() <= 2
            and all(value in [0, 1] for value in df[f].dropna().unique())
        ]
    binary_features = features_bool + features_bin_not_bool
    return binary_features


def get_all_binary_features(
    train_name, test_name, input_dir=DATA_INTERIM, verbose=True
):
    train_path = os.path.join(input_dir, train_name)
    train = cudf.read_csv(train_path)
    test_path = os.path.join(input_dir, test_name)
    test = cudf.read_csv(test_path)[train.columns]
    all_data = cudf.concat([train, test])
    binary_features = get_binary_features(all_data)

    if verbose:
        n_binary_features = len(binary_features)
        n_features = len(
            [f for f in train.columns if f not in ["TARGET", "SK_ID_CURR"]]
        )
        print(
            f"{n_binary_features} features sur {n_features} sont booléennes (ratio : {n_binary_features / n_features:.1%})"
        )
    return binary_features


# En fait entre pandas et cudf,
# c'est juste le get_indexer() qui ne fonctionne pas (sans être remonté dans la stack d'erreur)
# Il faut donc convertir en pandas
def cu_build_folds_list(n_folds, X, y, random_state=VAL_SEED, frac_test=0.25):
    # Si X nest pas un RAPIDS, on sort tout de suite
    if not isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        raise ValueError(
            f"Fonction 'cu_build_folds_list' : 'X' n'est pas un cudf.df, type réel : {type(X)}"
        )
        return

    # Si on utilise pas la validation croisée (temps de calculs trop longs)
    if n_folds == 1:
        X_train, X_val, y_train, y_val = sk_train_test_split(
            X.to_pandas(),
            y.to_pandas(),
            stratify=y.to_pandas(),
            test_size=frac_test,
            random_state=random_state,
        )

        # On convertit les indices en indices positionnels relatifs à X,
        # En réalité get_indexer() non supportée pour l'insant par RAPIDS
        train_idx = X.index.to_pandas().get_indexer(X_train.index)
        valid_idx = X.index.to_pandas().get_indexer(X_val.index)
        folds_list = [(train_idx, valid_idx)]

    # Si on utilise la validation croisée:
    else:
        folds = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state,
        )
        folds_list = [
            (train_idx, valid_idx)
            for train_idx, valid_idx in folds.split(X.to_pandas(), y.to_pandas())
        ]

    return folds_list


# Retourne les indices issus d'un train_test_split pour cudf ou pd.df mais en relatifs et non positionnels,
# C'est à dire sous la même forme qu'un folds ou kfolds
def one_k_fold(X, y, test_size=0.25, shuffle=True, random_state=VAL_SEED):

    # Si on est en RAPIDS, train_test_split n'accepte pas les NaN
    if isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        # On trouve 2 prédicteurs sans nan dans X (car pas de NaN pour cudf)
        columns_without_nan = [c for c in X.columns if X[c].isna().sum() == 0]
        if len(columns_without_nan) < 2:
            raise ValueError(
                f"La table Application doit comporter au moins 2 colonnes sans NaN"
            )
            return
        # Si on a suffisemment de colonnes sans nan, on effectue le split sur 2 colonnes pour récupérer les index des splits
        X_train_tmp, X_test_tmp, y_train, y_test = (
            cuml.model_selection.train_test_split(
                X[columns_without_nan[:2]],
                y,
                test_size=test_size,
                shuffle=shuffle,
                stratify=y,
                random_state=random_state,
            )
        )

    # Si les données d'entrée ne sont pas dans un cudf.df on consière que c'est un pandas dataframe
    X_train_tmp, X_test_tmp, y_train, y_test = sk_train_test_split(
        X[:2],
        y,
        test_size=test_size,
        shuffle=shuffle,
        stratify=y,
        random_state=random_state,
    )

    # On convertit les indices "absolus" en indices relatifs positionnels par rapport à X
    train_idx = X.index.get_indexer(X_train_tmp.index)
    valid_idx = X.index.get_indexer(X_test_tmp.index)
    folds_list = [(train_idx, valid_idx)]
    return folds_list


def train_test_split_nan(df, y=None, test_size=0.25, shuffle=True, random_state=42):
    """Réalise un train_test_split stratifié sur cudf ou pandas Dataframe, même s'il y a des NaN

    Args:
        df (cudf ou pandas DataFrame): Données à splitter
        y (cudf ou pandas Series, optional): target utilisée pour stratifier. Si elle n'est pas indiquée, il s'agit d'une colonne de df nommée 'TARGET'. Defaults to None.
        test_size (float, optional): fraction pour le jeu de test. Defaults to 0.25.
        shuffle (bool, optional): Mélange du dataset avant le split. Defaults to True.
        random_state (int, optional): Graine pour la reproductibilité des résultats. Defaults to 42.

    Returns:
        cudf ou pandas DF/Series selon le type de df: X_train, X_test, y_train, y_test
    """
    # Si y n'est pas fournie, on considère qu'il s'agit de la colonne nommée "TARGET" dans le df
    if y is None:
        if "TARGET" not in df.columns:
            print("TARGET n'est pas dans le dataset, veuillez préciser y")
            return
        else:
            y = df["TARGET"]
            X = df.drop("TARGET", axis=1)
    # Si la target a été fournie en param, on n'enlève pas 'TARGET' du df
    else:
        X = df

    # Si le dataset est un cudf
    if isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        if test_size == 0:
            return X, None, y, None
        n_missing = df.isna().sum().sum()
        # train_test_split de cuml n'autorise pas les nan
        # Si'il n'y a pas de nan, on utilise train_test_split de cuml
        if n_missing == 0:
            return cuml.model_selection.train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=shuffle,
                stratify=y,
                random_state=random_state,
            )
        # s'il y a des nan dans le cudf, il nous faut au moins 2 colonnes sans nan dans X
        else:
            if isinstance(X, cudf.core.series.Series):
                print(
                    "Le dataset doit comporter au moins 2 colonnes sans NaN or X est une série"
                )
                return
            columns_without_nan = [c for c in X.columns if X[c].isna().sum() == 0]
            if len(columns_without_nan) < 2:
                print("Le dataset doit comporter au moins 2 colonnes sans NaN")
                return
            # Si on a suffisemment de colonnes sans nan, on effectue le split sur 2 colonnes pour récupérer les index des splits
            # Attention à ne pas passer par un index pandas, via sklearn, car ce ne sont pas les mêmes indexers
            X_train_tmp, X_test_tmp, y_train, y_test = (
                cuml.model_selection.train_test_split(
                    X[columns_without_nan[:2]],
                    y,
                    test_size=test_size,
                    shuffle=shuffle,
                    stratify=y,
                    random_state=random_state,
                )
            )
            X_train = X.loc[X_train_tmp.index, X.columns]
            X_test = X.loc[X_test_tmp.index, X.columns]
            return X_train, X_test, y_train, y_test

    # Si le dataset n'est pas de type cudf, on considère qu'il s'agit d'un pandas df ou un numpy array
    # if isinstance(df, (pd.core.frame.DataFrame, pd.core.frame.Series))
    else:
        return sk_train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            stratify=y,
            random_state=random_state,
        )


def balance_nearmiss(
    X,
    y,
    k_neighbors=3,
    minority_to_majority_ratio=1.0,
    drop_null_var=False,
    verbose=True,
):

    check_no_nan(X)

    not_predictors = [
        f for f in X.columns if f in ["SK_ID_CURR", "TARGET", "Unnamed: 0"]
    ]
    if not_predictors:
        print(f"WARNING {not_predictors} ne doivent pas figurer dans les colonnes de X")
    if isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        y_ = y.to_pandas()
        X_ = X.to_pandas()
    else:
        # La copie est nécessaire pour pouvoir paralléliser les jobs
        y_ = y.copy()
        X_ = X.copy()

    # Le ratio 1 correspond à l'équilibre parfait
    if minority_to_majority_ratio == 1.0:
        sampling_strategy = "auto"
    else:
        n_samples_1_before = np.sum(y_)
        n_samples_0_before = len(y_) - n_samples_1_before
        ratio_before = n_samples_1_before / n_samples_0_before
        # Si le ratio fourni est plus grand que l'actuel, on diminue la classe majoritaire
        if minority_to_majority_ratio >= ratio_before:
            n_samples_0_after = int(
                np.round(n_samples_1_before / minority_to_majority_ratio, decimals=0)
            )
            sampling_strategy = {0: n_samples_0_after}
        # Si le ratio fourni est plus petit que l'actuel, on diminue encore plus la classe minoritaire
        else:
            n_samples_1_after = int(
                np.round(n_samples_0_before * minority_to_majority_ratio, decimals=0)
            )
            sampling_strategy = {1: n_samples_1_after}
    nm = NearMiss(
        n_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy,
    )
    X_resampled, y_resampled = nm.fit_resample(X_, y_)

    if isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        y_resampled = cudf.from_pandas(y_resampled)
        X_resampled = cudf.from_pandas(X_resampled)

    if verbose:
        print(
            f"Avant rééquilibrage : Nb observations = {X.shape[0]:_}, ratio défauts = {y.value_counts(normalize=True)[1]:.2%}"
        )
        print(
            f"Après NEARMISS sampling_strategy {sampling_strategy}: Nb observations = {X_resampled.shape[0]:_}, ratio défauts = {y_resampled.value_counts(normalize=True)[1]:.2%}"
        )

    # On vérifie les variables de variance nulle
    features_null_var = check_variances(X_resampled, raise_error=False, verbose=verbose)
    if drop_null_var:
        X_resampled = X_resampled.drop(features_null_var, axis=1)
    return X_resampled, y_resampled, features_null_var


# Bon article (en français !) sur SMOTE : https://kobia.fr/imbalanced-data-smote/
# Sur CUDA :
# il peut y avoir une erreur d'allocation 'pinned-memory' :
# L'algo effectue des transfers de mem cpu vers gpu, cela exige de la syncronisation CPU/GPU,
# (contrairement à de la mem qui reste sur GPU) donc de la 'pinned memory'.
# Les ressources en pinned memory sont limitées et sont partagées avec les autres applications.
# (une appli peut en réserver pour soi-même en oubliant les autres sans nettoyer,
# fonctionne par pooling qu'on peut s'octroyer)
# Plus de détail : https://docs.cupy.dev/en/stable/user_guide/memory.html
# ==> Si tu vois l'erreur 'pinned memory',
# ferme toutes les autres applis, et si cela ne suffit pas, redémarre.
# Le sampling_strategy=1 (pour arriver à un rééquilibragee default_ratio=50%)
# n'est pas la cause de l'erreur/warning 'pinned_mem',
# Toutefois + on augmente le % + c'est looooong. (vraiment très long !)
def balance_smote(
    X,
    y,
    k_neighbors=3,
    # On ne rééquilibre pas complètement pour limiter le tempsde traitement.
    sampling_strategy=0.7,
    random_state=42,
    binary_features=[],
    add_to_categorical=["AGE_RANGE"],
    discrete_features=[],
    verbose=False,
):
    """Rééquilibre un df/cudf avec de l'OverSampling SMOTE en fonction de 'TARGET'

    Args:
        X (df ou cudf): DataFrame contenant les données
        y (pd.Series ou cudf.Series): Série contenant la target
        binary_features (list) : Liste de features booléennes, considérées comme catégorielles.
        add_to_categorical (list): Liste de features à considérer comme catégorielles en plus des features booléennes issues du OneHot.
        random_state (int, optional): Graine de hasard pour la reproductibilité des résultats. Defaults to 42.

    Returns:
        (df ou cudf): le DataFrame rééquilibré en nombre d'observations
    """
    # Finalement, pour gagner en temps de calcul on ne vérifie pas l'absence de NaN.
    # check_no_nan(X)

    pinned_mempool = cp.get_default_pinned_memory_pool()
    pinned_mempool.free_all_blocks()
    gc.collect()

    # L'utilisation de cuml pour les plus proches voisins améliore nettement les temps de calculs
    # On enlève les warnings car sinon "Unused keyword parameter: n_jobs during cuML estimator initialization" dans optuna alors que n_jobs=1
    cuml.common.logger.set_level(2)
    cu_nn = NearestNeighbors(n_neighbors=k_neighbors)
    features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]

    # La copie est imérative pour pouvoir paralléliser les jobs en mode CPU
    y_ = y.to_numpy(copy=True)
    X_ = X[features].to_numpy(copy=True)
    # print("\ndtype X")
    # print(X[features].dtypes.value_counts())

    recast_categorical = False
    if not binary_features:
        features_bool = get_binary_features(X[features])
    else:
        features_bool = [f for f in binary_features if f in features]

    _add_to_categorical = [f for f in add_to_categorical if f in features]
    categorical_features = features_bool + _add_to_categorical
    # print("categorical_features", categorical_features)
    if categorical_features:
        recast_categorical = True
        categorical_id = [
            X[features].columns.get_loc(name) for name in categorical_features
        ]

        # Contrairement à SMOTE, SMOTENC ne rajoutera pas des valeurs floats intermédiaires pour les variables categorical
        # Cependant, le traitement est nettement plus long
        smote = SMOTENC(
            categorical_features=categorical_id,
            k_neighbors=cu_nn,
            # k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            # n_jobs=1,
        )
    else:
        recast_categorical = False
        smote = SMOTE(
            k_neighbors=cu_nn,
            # k_neighbors=k_neighbors,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    X_resampled, y_resampled = smote.fit_resample(X_, y_)
    # Si au départ on a un cudf on renvoie un cudf, sinon on renvoie un ps.Dataframe
    if isinstance(X, (cudf.core.dataframe.DataFrame, cudf.core.series.Series)):
        df_X = cudf.DataFrame(X_resampled, columns=features)
        df_y = cudf.Series(y_resampled)
    else:
        df_X = pd.DataFrame(X_resampled, columns=features)
        df_y = pd.Series(y_resampled)
        # print("dtypes df_X")
        # print(df_X.dtypes.value_counts())
        # print("valeurs manquantes dans df_X")
        # print(df_X.isna().sum().sum())
        # print(df_X.head())
        dtypes_before_smote = X[features].dtypes
        df_X = df_X.astype(dtypes_before_smote)
        # print("dtypes df_X après astype")
        # print(df_X.dtypes.value_counts())
    df_y.name = "TARGET"

    if recast_categorical:
        # On recaste les variables catégorielles qui sont devenues des float64
        df_X[features_bool] = df_X[features_bool].astype("bool")
        df_X[_add_to_categorical] = df_X[_add_to_categorical].astype("int16")

    # On arrondit les variables numériques discrètes à l'entier le plus proche (exemple CNT_CHILDREN)
    # Ne pas utiliser si ces variables ont été scalées et devenues floats
    if discrete_features:
        discrete_features_ = [f for f in discrete_features if f in X.columns]
        discrete_id = [X[features].columns.get_loc(name) for name in discrete_features_]
        for f in discrete_features:
            df_X[f] = np.round(df_X[f], decimals=0)
        df_X[discrete_features_] = df_X[discrete_features].astype(int)

    # Si l'index était dans les colonnes, on le repositionne. Les nouvelles lignes ont un index manquant
    if "SK_ID_CURR" in X.columns:
        df_X["SK_ID_CURR"] = X["SK_ID_CURR"]
        # On remplit les nouveaux index (manquants) par une valeur incrémentale
        start_value = 900_000
        missing_indices = df_X["SK_ID_CURR"].isna()
        increment_values = np.arange(start_value, start_value + missing_indices.sum())
        df_X.loc[missing_indices, "SK_ID_CURR"] = increment_values

    if verbose:
        print(
            f"Avant rééquilibrage : Nb observations = {y.shape[0]:_}, ratio défauts = {y.value_counts(normalize=True)[1]:.2%}"
        )
        """print(
            f"Après SMOTE : Nb observations = {df_y.shape[0]:_}, ratio défauts = {df_y.value_counts(normalize=True)[1]:.2%}"
        )"""
        print(
            f"Après SMOTE : Nb observations = {len(y_resampled):_}, ratio défauts = {np.mean(y_resampled):.2%}"
        )
    gc.collect()
    return df_X, df_y
