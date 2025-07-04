import datetime
import time
import gc
import os
import joblib

# import random
import numpy as np
import pandas as pd

# import os
# import torch
import matplotlib.pyplot as plt
from contextlib import contextmanager
import re
import cupy as cp
from shapely import length

import cudf

from src.p7_constantes import MODEL_DIR, DATA_CLEAN_DIR


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    duration = format_time(time.time() - t0)
    print(f"{name} - duration (hh:mm:ss) : {duration}")


# Fonction reprise de https://mccormickml.com/2019/07/22/BERT-fine-tuning/#31-bert-tokenizer
def format_time(duration_seconds):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    duration_rounded = int(round((duration_seconds)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=duration_rounded))


def format_time_min(duration_seconds):
    """
    Takes a time in seconds and returns a string hh:mm
    """
    # Arrondit au nombre de minutes le plus proche
    duration_rounded = int(round(duration_seconds / 60))

    # Calcul des heures et des minutes restantes
    hours = duration_rounded // 60
    minutes = duration_rounded % 60

    # Retourne au format hh:mm
    return "{:02d}:{:02d}".format(hours, minutes)


def reduce_memory_cudf(df):
    """Réduit l'utilisation mémoire d'un DataFrame cuDF."""
    # Adaptation du code pour cuDF
    # df.loc[:, col] = ... est moins efficace avec cuDF. Il est préférable d’utiliser df[col]
    # df.memory_usage() existe, mais pas toujours avec les mêmes options.
    # object n’est pas utilisé dans cuDF (il n’y a pas de colonnes mixtes de strings ou objets arbitraires)
    # Certaines conversions peuvent ne pas être supportées directement (comme vers float16)
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "bool":  # rien à faire
            continue

        try:
            c_min = df[col].min()
            c_max = df[col].max()
        except Exception as e:
            print(f"Impossible d'évaluer min/max pour {col}: {e}")
            continue

        if cp.issubdtype(col_type, cp.integer):
            """# On n'a plus de int8, ce sont des bool
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype('int8')
            #int16 sera moins performant que int32 sur GPU et va nous gêner dans les algorithmes cuML (conversion en interne car + performant ou obligation de caster)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype('int16')"""

            if c_min >= cp.iinfo(np.int8).min and c_max <= cp.iinfo(cp.int32).max:
                df[col] = df[col].astype("int32")
            else:
                df[col] = df[col].astype("int64")

        elif cp.issubdtype(col_type, cp.floating):
            # float16 est parfois moins supporté ou moins précis en GPU (donc élimine les float16)
            if c_min >= cp.finfo(cp.float16).min and c_max <= cp.finfo(cp.float32).max:
                df[col] = df[col].astype("float32")
            else:
                df[col] = df[col].astype("float64")

        # Pour les strings, cuDF ne permet pas de conversion directe pour réduire la mémoire
        # object n’est pas utilisé dans cuDF (il n’y a pas de colonnes mixtes de strings ou objets arbitraires)
        """elif np.issubdtype(col_type, np.object_):
            pass  # skip, cuDF strings restent en strings"""

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


# Sauvegarde un objet sous forme de fichier .pkl sans vérifier s'il existe déjà
def save_pkl(object, filename, directory=MODEL_DIR, title="", verbose=True):
    path = os.path.join(directory, f"{filename}.pkl")
    joblib.dump(object, path)
    if verbose:
        try:
            length_obj = len(object)
            length_str = f"(len = {length_obj}) "
        except:
            length_str = ""
        if title:
            title = title + " "
        else:
            title = f"Objet {object.__class__.__name__} "
        print(f"{title}{length_str}engistré(e) dans {path}")
    return


def read_pkl(filename, directory=MODEL_DIR, verbose=True):
    path = os.path.join(directory, f"{filename}.pkl")
    object = joblib.load(path)
    if verbose:
        print(f"Objet {object.__class__.__name__} lu dans {path}")
    return object


# Désactive la validation croisée
class DisabledCV:
    def __init__(self):
        self.n_splits = 1

    def split(self, X, y, groups=None):
        yield (np.arange(len(X)), np.arange(len(y)))

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


def clean_ram(list_vars, dic_vars):

    plt.close("all")

    vars_to_del = [var for var in list_vars if var in dic_vars.keys()]
    for var in vars_to_del:
        del dic_vars[var]
    gc.collect()
    print(f"{len(vars_to_del)} variables détruites : {vars_to_del}")
    return


def print_dtype_with_mixt(df):
    # Imprime les types de données d'un DataFrame même s'il s'agit d'un type de données mixte
    for column in df.columns:
        print(column, ":", pd.api.types.infer_dtype(df[column]))


# Retourne la liste des features booléennes (de type bool ou bool_).
# Si on convertit un np.array en cuDF, le type n'est pas np.bool mais np.bool_ (codé sur 8 bits = 1 byte)
# Nativement, les cuDF (et df de pandas) utilisent également des bool codés sur 8 bits
def get_bool_features(df, exclude=["SK_ID_CURR", "TARGET"], verbose=True):
    features = [col for col in df.columns if col not in exclude]
    bool_features = [
        col for col in df[features] if str(df[col].dtype).startswith("bool")
    ]
    if verbose:
        if exclude:
            add_exclude = f" en excluant {exclude}"
        print(f"{len(bool_features)} features booléennes{add_exclude}")
        print(bool_features)
    return bool_features


def ask_confirmation(
    message="Voulez-vous continuer (y/o pour continuer) ?", verbose=True
):
    response = input(f"{message} : ").strip().lower()
    if response and (response == "y" or response == "o"):
        confirmed = True
    else:
        confirmed = False
    if verbose:
        if confirmed:
            print("Action confirmée")
        else:
            print("Action abandonnée")
    return confirmed


def search_item_regex(regex, search_in, case_sensitive=False, verbose=True):
    """
    Sélectionne les éléments d'une collection qui correspondent à une expression régulière.

    Args:
        regex (str): Expression régulière à rechercher.
        search_in (list[str] | Index | set[str]): Collection de chaînes à parcourir.
        ignore_case (bool, optional): Si True, ignore la casse (par défaut: True).
        verbose (bool, optional): Si True, affiche les résultats trouvés (par défaut: True).

    Returns:
        list[str] : Liste ou ensemble des éléments correspondants à l'expression.

    Aide - Rappels utiles sur les expressions régulières :
        ^    : Début de chaîne
        $    : Fin de chaîne
        .    : N'importe quel caractère (sauf retour à la ligne)
        *    : Répète 0 ou plusieurs fois
        +    : Répète 1 ou plusieurs fois
        ?    : Rend l'élément précédent optionnel
        []   : Classe de caractères (ex: [abc] pour a, b ou c)
        |    : OU logique (ex: a|b)
        ()   : Regroupe des motifs
        \\   : Échappe un caractère spécial (ex: \\ pour un antislash, \\. pour un point)

    Exemple :
        Extension de fichier
        >>> search_item_regex(r"\\.csv$", ["data.csv", "image.png", "notes.csv"])
        ['data.csv', 'notes.csv']
    """
    # ... implémentation ...
    if case_sensitive:
        pattern = re.compile(regex)
    else:
        pattern = re.compile(regex, re.IGNORECASE)
    items_matched = []

    if search_in is not None:
        for item in search_in:
            if isinstance(item, str):
                if pattern.search(item):
                    items_matched.append(item)

    if verbose:
        print(
            f"{len(items_matched)} éléments correspondent au motif '{regex}' (Respecter la casse : {case_sensitive}):"
        )
        print(items_matched)

    return items_matched


def read_train(
    directory=DATA_CLEAN_DIR,
    train_name="01_v2_vif_train.csv",
    features=[],
    optim=True,
    clean_dtype=False,
):
    if features:
        train = cudf.read_csv(os.path.join(directory, train_name))[
            features + ["SK_ID_CURR", "TARGET"]
        ].set_index("SK_ID_CURR")
    else:
        train = cudf.read_csv(os.path.join(directory, train_name)).set_index(
            "SK_ID_CURR"
        )
    if optim:
        train = reduce_memory_cudf(train)
    predictors = [f for f in train.columns if f not in ["SK_ID_CURR", "TARGET"]]
    X = train[predictors].to_pandas()
    # Un modèle enregistré dans le registre mlflow, ne doit pas être entraîné sur des données de type intqui comportent des NaN.
    # Par ailleurs, pour une analyse SHAP, les booléens peuvent nous poser des difficultés.
    # Le plus simple est de transformer les int en float, puis de transformer les booéens en int.
    if clean_dtype:
        print("Cast des features")
        int_features = X.select_dtypes(include="int").columns.to_list()
        if int_features:
            X[int_features] = X[int_features].astype("float")
        bool_features = X.select_dtypes(include="bool").columns.to_list()
        if bool_features:
            X[bool_features] = X[bool_features].astype("int")
    y = train["TARGET"].to_pandas()
    print("\nInfo X_train")
    X.info()
    return X, y
