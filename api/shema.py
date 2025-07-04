# src/models/input_data.py pour checker le format d'input des données envoyées dans la requête
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Union


# La réponse à un get_features contiendra le client_id et les features dans un dictionnaire
class FeaturesResponse(BaseModel):
    client_id: int
    features: Dict[str, Any]


# Pour envoyer une requête de demande de prédiction, le ID client doit être un int
class PredictionRequestOld(BaseModel):
    # client_id: int = Field(..., example=1234)
    client_id: int = Field(...)

    # Remplace le mot clef example dans Field() car c'est deprecated
    model_config = ConfigDict(json_schema_extra={"example": {"client_id": 258223}})


class PredictionRequest(BaseModel):
    client_id: int = Field(...)
    features: dict[str, Union[float, int, None]] = Field(...)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


# Shéma de réponse à une requête post predict
class PredictionResponse(BaseModel):
    client_id: int
    features: Dict[str, Any]
    proba: float
    prediction: int
    threshold: float


# Shéma de réponse à une requête post predict_proba
class ProbaResponse(BaseModel):
    client_id: int
    proba: float
