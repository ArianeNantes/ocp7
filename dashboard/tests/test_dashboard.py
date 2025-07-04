import pytest
import os
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DTYPES_NAME = "dtypes_lgbm.pkl"


from dashboard.dashboard import load_explainer, load_new_loan_ids, load_train


##################################################### Test chargement des données


# On fait une ressource (une fixture) car on va s'en resservir
@pytest.fixture
def expected_dtypes():
    expected_dtypes = joblib.load(os.path.join(DATA_DIR, DTYPES_NAME))
    return expected_dtypes


# Test du chargement de l'explainer
def test_load_explainer(expected_dtypes):
    explainer = load_explainer()
    assert explainer is not None, "L'explainer n'est pas chargé"

    # L'explainer contient bien une valeur de base et des shap_values
    assert (
        explainer.expected_value is not None
    ), "L'explainer ne contient pas d'expected_value"
    assert (
        explainer.shap_values is not None
    ), "L'explainer ne contient pas de shap_values"

    # L'explainer contient des informations sur les bonnes features
    expected_features = list(expected_dtypes.keys())
    explainer_features = explainer.feature_names
    assert set(explainer_features) == set(
        expected_features
    ), "Les features de l'explainer ne correspondent pas à celles du jeu de test"

    # La prédiction par l'explainer est proche de celle effectuée par le modèle
    train, features = load_train()
    # On prend le premier id qu'on trouve dans le train
    loan_id = train.head(1).index[0].astype(int)
    # La probabilité calculée par le modèle figure déjà dans le fichier
    model_proba = train["PROBA"].loc[loan_id]
    # On calcule les shap values du client de train concerné
    shap_values_loan = explainer(train.loc[[loan_id], features])
    # On calcule la sortie calculée par l'explainer
    explainer_prediction = explainer.expected_value + shap_values_loan.values.sum()
    # On compare les sorties de l'explainer et du modèle à 4 chiffres après la virgule
    assert round(model_proba, 4) == round(
        explainer_prediction, 4
    ), "La sortie de l'explainer est trop éloignée de la prédiction faite par le modèle"


# Test du chargement des données représentant la population d'entraînement
def test_load_train(expected_dtypes):
    train, features = load_train()

    # 1. Vérifie que les colonnes attendues sont là
    missing_cols = set(expected_dtypes.keys()) - set(train[features].columns)
    assert not missing_cols, f"Colonnes manquantes : {missing_cols}"

    # 2. Vérifie les dtypes (on teste juste le nom du type ici)
    for col, expected_type in expected_dtypes.items():

        if col in train[features].columns:
            actual_type = str(train[col].dtype)
            assert (
                actual_type == expected_type
            ), f"{col}: attendu {expected_type}, obtenu {actual_type}"

    # 3. Pas de bool
    bool_cols = [col for col in train[features].columns if train[col].dtype == bool]
    assert not bool_cols, f"Colonnes bool trouvées : {bool_cols}"

    # 4. Train contient bien les colonnes PROBA et PREDICTION
    assert (
        "PROBA" in train.columns
    ), "Le fichier de train ne contient pas la colonne 'PROBA'"
    assert (
        "PREDICTION" in train.columns
    ), "Le fichier de train ne contient pas la colonne 'PREDICTION'"


# Test de chargement des ID des nouveaux clients
def test_load_new_loan_ids():
    ids = load_new_loan_ids()
    train, _ = load_train()
    train_ids = train.index.to_list()

    # La liste n'est ni None ni vide
    assert ids is not None, "La liste des nouveaux emprunts est None"
    assert len(ids) > 0, "La liste des nouveaaux emprunts est vide"

    # Le client id 0 n'existe pas
    assert 0 not in ids, "L'ID 0 ne doit pas figurer dans la liste des nouveaux IDs"

    # Les fichiers train et la liste des nouveaux id n'ont aucun id en commun
    assert not set(ids).intersection(
        train_ids
    ), "Les données de train et de nouveaux emprunts ont des IDs en commun"
