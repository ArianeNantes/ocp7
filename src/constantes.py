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

VAL_SEED = 42

# Tolérance numérique
EPSILON = 1e-8

# Informations sur le meilleur modèle (enregistrées dans MODEL_DIR)
BEST_MODEL_NAME = "best_model_lgbm.pkl"
BEST_THRESHOLD_NAME = "best_threshold_lgbm.pkl"
BEST_DTYPES_NAME = "dtypes_lgbm.pkl"
EXPLAINER_NAME = "explainer_lgbm.pkl"
GLOBAL_IMPORTANCES_NAME = "global_importances.csv"

# Nom du fichier contenant les nouveaux emprunts (en provenance de application_test)
NEW_LOANS_NAME = "X_new_loans_lgbm.csv"
# Nom du fichier échantillon représentatif de la population d'entrainement
SAMPLE_TRAIN_NAME = "sampled_train_lgbm.csv"
