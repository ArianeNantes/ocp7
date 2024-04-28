import pandas as pd
import numpy as np
from IPython.display import display


# URL de téléchargement des données
DATA_URL = "https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip"

# Nom du fichier à télécharger
FILE_ZIP = "Projet+Mise+en+prod+-+home-credit-default-risk.zip"

DATA_DIR = "data/"
# Répertoire pour les données d'origines
DATA_BASE = "data/base/"
# Répertoire pour sauvegarder les data travaillées (sauvegardes intermédiaires)
DATA_INTERIM = "data/interim/"
# Répertoire pour sauvegarder les données complètement nettoyées
DATA_CLEAN_DIR = "data/cleaned/"

PRETRAINED_MODEL_DIR = "pretrained_model/"

URI_LOCAL = "http://127.0.0.1:8080"

# Répertoires contenant les images de train et de test
IMG_TRAIN = "data/cleaned/img_train/"
IMG_TEST = "data/cleaned/img_test/"
IMG_VAL = "data/cleaned/img_val/"

MODEL_DIR = "models/"


# Style et palette de base SNS pour les plots
STYLE = "whitegrid"  # Style de base pour graphiques sns
PALETTE_HIGH = "cool"  # graphiques avec cmap - différences de couleurs intenses
PALETTE_LOW = "muted"  # Diagrammes en barres etc.

# GENERAL CONFIGURATIONS
NUM_THREADS = 8
# DATA_DIRECTORY = "../input/"
DATA_DIRECTORY = "data/base/"
SUBMISSION_SUFIX = "_model2_04"
# INSTALLMENTS TREND PERIODS
INSTALLMENTS_LAST_K_TREND_PERIODS = [12, 24, 60, 120]

# LIGHTGBM CONFIGURATION AND HYPER-PARAMETERS
GENERATE_SUBMISSION_FILES = True
STRATIFIED_KFOLD = False
RANDOM_SEED = 737851
NUM_FOLDS = 10
# NUM_FOLDS = 5
EARLY_STOPPING = 100
# EARLY_STOPPING = 50

# [TODO] Modifier ça , ce ne sont pas des constantes
LIGHTGBM_PARAMS = {
    "boosting_type": "goss",
    "n_estimators": 10000,
    "learning_rate": 0.005134,
    "num_leaves": 54,
    "max_depth": 10,
    "subsample_for_bin": 240000,
    "reg_alpha": 0.436193,
    "reg_lambda": 0.479169,
    "colsample_bytree": 0.508716,
    "min_split_gain": 0.024766,
    "subsample": 1,
    "is_unbalance": False,
    "silent": -1,
    "verbose": -1,
}
