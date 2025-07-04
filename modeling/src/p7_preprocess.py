import cudf
import cuml
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from sklearn.model_selection import train_test_split as sk_train_test_split

"""
import numpy as np
import pandas as pd
import gc
import os
from copy import deepcopy
import cupy as cp
import sklearn
from cuml.preprocessing import StandardScaler

from cuml.linear_model import LinearRegression as CuLinearRegression
from cuml.metrics import r2_score
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.impute import SimpleImputer as skSimpleImputer
from sklearn.preprocessing import RobustScaler as skRobustScaler
from sklearn.preprocessing import MinMaxScaler as skMinMaxScaler
from sklearn.feature_selection import VarianceThreshold as skVarianceThreshold

from modeling.src.p7_constantes import VAL_SEED, DATA_INTERIM
"""


import warnings


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


def check_no_nan(df):
    features_with_nan = [f for f in df.columns if df[f].isna().sum() > 0]
    if features_with_nan:
        raise ValueError(
            f"Eliminez d'abord les NaN, {len(features_with_nan)} features ont des valeurs manquantes :\n{features_with_nan}"
        )


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
