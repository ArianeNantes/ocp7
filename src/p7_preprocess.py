import numpy as np

import cudf
import cuml
import sklearn
from cuml.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, SMOTENC

from copy import deepcopy

# from sklearn.cluster import AgglomerativeClustering
# from collections import defaultdict


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

    def __init__(self, threshold=0, verbose=False):
        self.threshold = threshold
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
        if self.to_drop_:
            # On supprime les colonnes de variances trop faibles
            X_ = X.drop(self.to_drop_, axis=1, inplace=False)
        else:
            X_ = X
        self.feature_names_out_ = X_.columns.tolist()
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


class CuMinMaxScaler_old(BaseEstimator, TransformerMixin):
    """
    Scale les variables non booléennes avec un MinMaxScaler
    """

    def __init__(self, to_scale=[], feature_range=(0, 1), verbose=True):
        self.to_scale = to_scale
        self.feature_range = feature_range
        self.verbose = verbose
        self.scaler = MinMaxScaler(feature_range=self.feature_range)

    def fit(self, X, y=None):
        # Si to_scale n'a pas été spécifié, on mettra à l'échelle toutes les variables non booléennes
        if not self.to_scale:
            features = [f for f in X.columns if f not in ["SK_ID_CURR", "TARGET"]]
            features_not_bool = (
                X[features].select_dtypes(exclude="bool").columns.tolist()
            )
            # print("features_not_bool", features_not_bool)
            features_binary = [
                f
                for f in features_not_bool
                if X[f].dropna().nunique() <= 2
                and all(value in [0, 1] for value in X[f].dropna().unique().to_numpy())
            ]
            self.to_scale = [f for f in features_not_bool if f not in features_binary]

        # Si des varaiables à mettre à l'échelle ont des variances nulles, on lève une erreur
        check_variances(X[self.to_scale], raise_error=True)

        self.scaler.fit(X[self.to_scale].to_numpy())
        # Documentation RAPIDS : https://docs.rapids.ai/api/cuml/stable/api/#cuml.preprocessing.MinMaxScaler
        self.scale_ = self.scaler.scale_
        self.min_ = self.scaler.min_
        self.data_range_ = self.scaler.data_range_
        self.n_samples_seen_ = self.scaler.n_samples_seen_

        self.feature_names_in_ = X.columns.tolist()
        self.feature_names_out_ = []
        return self

    def transform(self, X, y=None, verbose=False):
        check_is_fitted(self)
        X_ = X.copy()
        scaled_features = cudf.DataFrame(
            self.scaler.transform(X_[self.to_scale].to_numpy()),
            columns=self.to_scale,
            index=X.index,
        )
        for feature in self.to_scale:
            X_[feature] = scaled_features[feature]
        self.feature_names_out_ = X_.columns.tolist()
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


def check_variances(df, raise_error=True):
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
            print(
                f"WARNING : {len(null_var)} variables de variance nulle :\n{null_var}"
            )
            return null_var
    else:
        return


def get_binary_features(df):
    features = [f for f in df.columns if f not in ["TARGET", "SK_ID_CURR"]]
    features_bool = df[features].select_dtypes(include="bool").columns.tolist()
    features_not_bool = [f for f in features if f not in features_bool]

    features_bin_not_bool = [
        f
        for f in features_not_bool
        if df[f].dropna().nunique() <= 2
        and all(value in [0, 1] for value in df[f].dropna().unique().to_numpy())
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
            # Si on a suffisemment de colonnes sans nan, on effetue le split sur 2 colonnes pour récupérer les index des slits
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
        return sklearn.model_selection.train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            stratify=y,
            random_state=random_state,
        )


def balance_nearmiss(
    df,
    random_state=42,
    verbose=True,
):
    """Rééquilibre un df/cudf avec de l'undersampling NearMiss

    Args:
        df (df ou cudf): DataFrame contenant les données y compris la target
        random_state (int, optional): Graine de hasard pour la reproductibilité des résultats. Defaults to 42.

    Returns:
        (df ou cudf): le DataFrame rééquilibré en nombre d'observations
    """
    check_no_nan(df)

    # Si la target n'est pas présente dans X, on sort car on en a besoin dans y
    if not "TARGET" in df.columns:
        print("La colonne 'TARGET' ne figure pas dans les données")
        return
    # On sépare la target des données (on laisse l'ID)
    features = [f for f in df.columns if f not in ["TARGET"]]
    y = df["TARGET"].to_numpy()
    X = df[features].to_numpy()

    nm = NearMiss()
    X_resampled, y_resampled = nm.fit_resample(X, y)

    cu_df = cudf.DataFrame(X_resampled, columns=features)
    cu_df["TARGET"] = y_resampled

    if verbose:
        print(
            f"Avant rééquilibrage : Nb observations = {df['TARGET'].shape[0]:_}, ratio défauts = {df['TARGET'].value_counts(normalize=True)[1]:.2%}"
        )
        print(
            f"Après NEARMISS : Nb observations = {cu_df['TARGET'].shape[0]:_}, ratio défauts = {cu_df['TARGET'].value_counts(normalize=True)[1]:.2%}"
        )
    # On vérifie les variables de variance nulle
    check_variances(cu_df, raise_error=False)

    return cu_df


def balance_smote(
    df,
    random_state=42,
    add_to_categorical=["AGE_RANGE"],
    verbose=True,
):
    """Rééquilibre un df/cudf avec de l'OverSampling SMOTE en fonction de 'TARGET'

    Args:
        df (df ou cudf): DataFrame contenant les données y compris la target
        strategy (str, optional): Stratégie de rééquilibrage : NearMiss ou SMOTE. Defaults to "nearmiss".
        random_state (int, optional): Graine de hasard pour la reproductibilité des résultats. Defaults to 42.

    Returns:
        (df ou cudf): le DataFrame rééquilibré en nombre d'observations
    """
    check_no_nan(df)

    # Si la target n'est pas présente dans X, on en a besoin dans y
    if not "TARGET" in df.columns:
        print("La colonne 'TARGET' ne figure pas dans les données")
        return

    # On sépare la target des données
    features = [f for f in df.columns if f not in ["SK_ID_CURR", "TARGET"]]
    y = df["TARGET"].to_numpy()
    X = df[features].to_numpy()

    recast_categorical = False
    features_bool = get_binary_features(df[features])
    categorical_features = features_bool + add_to_categorical
    if categorical_features:
        recast_categorical = True
        categorical_id = [
            df[features].columns.get_loc(name) for name in categorical_features
        ]

        # Contrairement à SMOTE, SMOTENC ne rajoutera pas des valeurs floats intermédiaires pour les variables categorical
        # Cependant, le traitement est nettement plus long
        smote = SMOTENC(categorical_features=categorical_id, random_state=random_state)
    else:
        recast_categorical = False
        smote = SMOTE(random_state=random_state)

    X_resampled, y_resampled = smote.fit_resample(X, y)
    cu_df = cudf.DataFrame(X_resampled, columns=features)

    if recast_categorical:
        # On recaste les variables catégorielles qui sont devenues des float64
        cu_df[features_bool] = cu_df[features_bool].astype("bool")
        cu_df[add_to_categorical] = cu_df[add_to_categorical].astype("int16")

    # On repositionne la target à l'intérieur du df
    cu_df["TARGET"] = y_resampled

    # On repositionne l'index. Les nouvelles lignes ont un index manquant
    cu_df["SK_ID_CURR"] = df["SK_ID_CURR"]
    # On remplit les nouveaux index (manquants) par une valeur incrémentale
    start_value = 900_000
    missing_indices = cu_df["SK_ID_CURR"].isna()
    increment_values = np.arange(start_value, start_value + missing_indices.sum())
    cu_df.loc[missing_indices, "SK_ID_CURR"] = increment_values

    if verbose:
        print(
            f"Avant rééquilibrage : Nb observations = {df['TARGET'].shape[0]:_}, ratio défauts = {df['TARGET'].value_counts(normalize=True)[1]:.2%}"
        )
        print(
            f"Après SMOTE : Nb observations = {cu_df['TARGET'].shape[0]:_}, ratio défauts = {cu_df['TARGET'].value_counts(normalize=True)[1]:.2%}"
        )
    return cu_df
