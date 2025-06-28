import pytest
from fastapi.testclient import TestClient


from main import app
import os
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from src.constantes import (
    DATA_CLEAN_DIR,
    BEST_DTYPES_NAME,
)
from src.api.predictor import load_model, load_data_test


##################################################### Test des routes
client = TestClient(app)


# Test de la requête GET qui retourne les données d'un client existant dans la base
def test_read_features_existing_client():
    response = client.get("/features/258223")
    assert response.status_code == 200
    json_data = response.json()

    # Vérifie les clés présentes
    assert "client_id" in json_data
    assert "features" in json_data
    assert json_data["client_id"] == 258223
    assert isinstance(json_data["features"], dict)
    assert len(json_data["features"]) > 0


# Test de la requête GET qui renvoie un message d'erreur si le client est inexistant dans la base
def test_read_features_unknown_client():
    response = client.get("/features/999999999")  # Un ID inexistant
    assert response.status_code == 404
    assert response.json()["detail"] == "Désolé, cet ID client n'est pas répertorié."


# Test de la requête POST qui à partir d'un client existant renvoie le predict_proba et la prédiction en fonction du seuil
def test_predict_existing_client():
    input_json = {
        "client_id": 258259,
        "features": {
            "CODE_GENDER": 0,
            "AMT_ANNUITY": 69385.5,
            "REGION_POPULATION_RELATIVE": 0.035792,
            "DAYS_BIRTH": -18434,
            "DAYS_EMPLOYED": None,
            "DAYS_REGISTRATION": -868,
            "DAYS_ID_PUBLISH": -1979,
            "OWN_CAR_AGE": None,
            "REGION_RATING_CLIENT_W_CITY": 2,
            "EXT_SOURCE_1": 0.46206376,
            "EXT_SOURCE_2": 0.6427265,
            "EXT_SOURCE_3": 0.21518241,
            "YEARS_BEGINEXPLUATATION_MODE": 0.998,
            "LIVINGAREA_MODE": 0.4123,
            "TOTALAREA_MODE": 0.3113,
            "DEF_30_CNT_SOCIAL_CIRCLE": 2,
            "DAYS_LAST_PHONE_CHANGE": -3135,
            "CREDIT_TO_ANNUITY_RATIO": 10.245542,
            "CREDIT_TO_GOODS_RATIO": 1.0461987,
            "ANNUITY_TO_INCOME_RATIO": 0.34264445,
            "INCOME_TO_BIRTH_RATIO": -10.985136,
            "NAME_EDUCATION_TYPE_Highereducation": 1,
            "NAME_FAMILY_STATUS_Married": 1,
            "BURO_DAYS_CREDIT_MAX": -174,
            "BURO_DAYS_CREDIT_ENDDATE_MEAN": 1501.091,
            "BURO_AMT_CREDIT_SUM_MAX": 531000,
            "BURO_AMT_CREDIT_SUM_DEBT_MEAN": 13871.637,
            "BURO_CREDIT_ACTIVE_Active_MEAN": 0.5,
            "BURO_CREDIT_TYPE_Microloan_MEAN": 0,
            "BURO_CREDIT_TYPE_Mortgage_MEAN": 0,
            "BURO_STATUS_0_MEAN_MEAN": 0.5197955,
            "ACTIVE_DAYS_CREDIT_MAX": -174,
            "ACTIVE_DAYS_CREDIT_ENDDATE_MIN": -581,
            "ACTIVE_DAYS_CREDIT_ENDDATE_MAX": 9797,
            "ACTIVE_DAYS_CREDIT_UPDATE_MEAN": -119.5,
            "ACTIVE_AMT_CREDIT_MAX_OVERDUE_MEAN": 3060,
            "ACTIVE_AMT_CREDIT_SUM_MEAN": 121906.875,
            "ACTIVE_AMT_CREDIT_SUM_SUM": 731441.25,
            "ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN": 22500,
            "ACTIVE_MONTHS_BALANCE_SIZE_MEAN": 22.333334,
            "CLOSED_DAYS_CREDIT_MAX": -952,
            "CLOSED_DAYS_CREDIT_VAR": 441636.8,
            "CLOSED_DAYS_CREDIT_ENDDATE_MAX": 9440,
            "CLOSED_DAYS_CREDIT_UPDATE_MEAN": -760.5,
            "CLOSED_AMT_CREDIT_MAX_OVERDUE_MEAN": 2146.5,
            "CLOSED_AMT_CREDIT_SUM_MEAN": 143427.75,
            "CLOSED_AMT_CREDIT_SUM_SUM": 860566.5,
            "PREV_AMT_ANNUITY_MIN": 2250,
            "PREV_APP_CREDIT_PERC_MAX": 1,
            "PREV_AMT_GOODS_PRICE_MIN": 45000,
            "PREV_HOUR_APPR_PROCESS_START_MAX": 16,
            "PREV_DAYS_DECISION_MIN": -1095,
            "PREV_DAYS_DECISION_MAX": -16,
            "PREV_CNT_PAYMENT_SUM": 118,
            "PREV_NAME_YIELD_GROUP_low_action_MEAN": 0.125,
            "APPROVED_AMT_ANNUITY_MAX": 46941.434,
            "APPROVED_APP_CREDIT_PERC_MIN": 0.8632597,
            "APPROVED_APP_CREDIT_PERC_MAX": 1.2504731,
            "APPROVED_AMT_DOWN_PAYMENT_MAX": 18000,
            "APPROVED_AMT_GOODS_PRICE_MAX": 1125000,
            "APPROVED_DAYS_DECISION_MAX": -62,
            "APPROVED_CNT_PAYMENT_MEAN": 14.75,
            "REFUSED_AMT_CREDIT_MIN": 90000,
            "REFUSED_APP_CREDIT_PERC_MIN": 1,
            "REFUSED_DAYS_DECISION_MAX": -316,
            "POS_MONTHS_BALANCE_MEAN": -22.736841,
            "POS_SK_DPD_DEF_MEAN": 0,
            "POS_COUNT": 38,
            "INSTAL_DPD_MEAN": 0,
            "INSTAL_DBD_MEAN": 4.6125,
            "INSTAL_DBD_SUM": 738,
            "INSTAL_PAYMENT_PERC_MEAN": 0.99375,
            "INSTAL_PAYMENT_PERC_SUM": 159,
            "INSTAL_PAYMENT_DIFF_MAX": 192.915,
            "INSTAL_PAYMENT_DIFF_MEAN": 1.40625,
            "INSTAL_PAYMENT_DIFF_SUM": 225,
            "INSTAL_AMT_INSTALMENT_MAX": 147150,
            "INSTAL_AMT_INSTALMENT_MEAN": 10658.891,
            "INSTAL_AMT_INSTALMENT_SUM": 1705422.5,
            "INSTAL_AMT_PAYMENT_MIN": 32.085,
            "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -13,
            "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -169819,
            "CC_AMT_BALANCE_MEAN": 5912.852,
            "CC_AMT_PAYMENT_CURRENT_SUM": 971718.2,
            "CC_CNT_DRAWINGS_ATM_CURRENT_MEAN": 0,
            "CC_CNT_DRAWINGS_CURRENT_MEAN": 2.1981132,
            "CC_CNT_DRAWINGS_CURRENT_VAR": 64.97942,
        },
    }

    response = client.post("/predict", json=input_json)
    assert response.status_code == 200
    data = response.json()

    assert "client_id" in data
    assert "proba" in data
    assert "prediction" in data
    assert "threshold" in data
    assert 0 <= data["proba"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["client_id"] == 258259
    assert data["threshold"] == 0.48


##################################################### Test chargement des données


# On fait une ressource (une fixture) car on va s'en resservir
@pytest.fixture
def expected_dtypes():
    expected_dtypes = joblib.load(os.path.join(DATA_CLEAN_DIR, BEST_DTYPES_NAME))
    return expected_dtypes


# Test du chargement des nouveaux clients
def test_load_data_test_dtypes(expected_dtypes):
    df_test = load_data_test()

    # 1. Vérifie que les colonnes attendues sont là
    missing_cols = set(expected_dtypes.keys()) - set(df_test.columns)
    assert not missing_cols, f"Colonnes manquantes : {missing_cols}"

    # 2. Vérifie les dtypes (on teste juste le nom du type ici)
    for col, expected_type in expected_dtypes.items():

        if col in df_test.columns:
            actual_type = str(df_test[col].dtype)
            assert (
                actual_type == expected_type
            ), f"{col}: attendu {expected_type}, obtenu {actual_type}"

    # 3. Pas de bool
    bool_cols = [col for col in df_test.columns if df_test[col].dtype == bool]
    assert not bool_cols, f"Colonnes bool trouvées : {bool_cols}"

    # 4. L'ID 258223 est présent
    assert 258223 in df_test.index, "L'ID 258223 est introuvable dans les données"


# Test du chargement du modèle
def test_load_model(expected_dtypes):

    model = load_model()
    assert model is not None, "Le modèle n'est pas chargé"

    # 1. Vérifie que le modèle est fitté
    check_is_fitted(model)

    # 2. Vérifie la cohérence entre les features d'entraînement et les features du jeu de test
    # On vérifie seulement le nom des features, pas leur type
    model_features = model.feature_name_ if hasattr(model, "feature_name_") else None
    expected_features = list(expected_dtypes.keys())

    assert (
        model_features is not None
    ), "Le modèle ne contient pas d'information sur les features"
    assert set(model_features) == set(
        expected_features
    ), "Les features du modèle ne correspondent pas à celles du jeu de test"
