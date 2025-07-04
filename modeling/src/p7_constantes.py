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

MODEL_DIR = "models/"

# Style et palette de base SNS pour les plots
STYLE = "whitegrid"  # Style de base pour graphiques sns
PALETTE_HIGH = "cool"  # graphiques avec cmap - différences de couleurs intenses
PALETTE_LOW = "muted"  # Diagrammes en barres etc.

# GENERAL CONFIGURATIONS
NUM_THREADS = 16

# LIGHTGBM CONFIGURATION AND HYPER-PARAMETERS
GENERATE_SUBMISSION_FILES = True
STRATIFIED_KFOLD = False
# VAL_SEED = 737851 * fro kernel
VAL_SEED = 42
NUM_FOLDS = 10
# NUM_FOLDS = 5
EARLY_STOPPING = 100
# EARLY_STOPPING = 50

# Taille max du df X pour une réelle parallélisation des threads (optuna)
MAX_SIZE_PARA = 640

PORT_MLFLOW = "5000"
HOST_MLFLOW = "127.0.0.1"

# Tolérance numérique
EPSILON = 1e-10
