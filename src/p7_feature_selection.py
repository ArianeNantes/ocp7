import numpy as np
import pandas as pd
import os
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split

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

from src.p7_constantes import DATA_INTERIM
from src.p7_simple_kernel import get_memory_consumed, reduce_memory


class DataSelector:
    def __init__(self, train, test, num_prj="00", output_dir=DATA_INTERIM) -> None:
        self.train = train
        self.test = test
        self.num_prj = num_prj
        self.output_dir = output_dir
        self.id = "SK_ID_CURR"
        self.target = "TARGET"
        self.not_predictors = ["TARGET", "SK_ID_CURR"]
        self.predictors = [
            f for f in self.train.columns.tolist() if f not in self.not_predictors
        ]
        self.resampling = ""

    # Elimine les features de variance nulle
    def drop_null_std(self, verbose=True):
        features = [
            feature
            for feature in self.train.columns
            if feature not in ["SK_ID_CURR", "TARGET"]
        ]
        all_std = self.train[features].std()
        null_std_features = all_std[all_std == 0.0].index.tolist()
        if null_std_features:
            self.train = self.train.drop(null_std_features, axis=1)
            self.test = self.test.drop(null_std_features, axis=1)
        if verbose:
            if null_std_features:
                print(
                    f"{len(null_std_features)} features de variance nulle suppimées dans Train et Test"
                )
                print(
                    f"Nouvelle taille du jeu de Train : {self.train.shape}, {get_memory_consumed(self.train, verbose=False)} Mo"
                )
            else:
                print(
                    f"Aucune feature de variance nulle. Taille du jeu de Train :  {self.train.shape}, {get_memory_consumed(self.train, verbose=False)} Mo"
                )
        return self.train, self.test

    def reduce_memory_usage(self, inplace=False):
        if not inplace:
            df = self.train.copy()
            reduce_memory(df)
            return df
        else:
            reduce_memory(self.train)
            reduce_memory(self.test)
            return

    def save_data(self, train_name, test_name, replace=False):
        train_path = os.path.join(self.output_dir, train_name)
        test_path = os.path.join(self.output_dir, test_name)
        save = True
        if not replace:
            if os.path.exists(train_path) or os.path.exists(test_path):
                if os.path.exists(train_path):
                    print(f"Le fichier {train_path} existe déjà")
                if os.path.exists(test_path):
                    print(f"Le fichier {test_path} existe déjà")
                print(
                    "Sauvegarde des données non effectuée. Modifiez les noms de fichier ou forcer avec repace=True"
                )
                save = False
            if save:
                self.train.to_csv(train_path)
                print(f"Le fichier {train_path} sauvegardé. Forme {self.train.shape}")
                self.test.to_csv(test_path)
                print(f"Le fichier {test_path} sauvegardé. Forme {self.test.shape}")

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
