# src/models/input_data.py pour checker le format d'input des données envoyées dans la requête
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any


# La réponse à un get_features contiendra le client_id et les features dans un dictionnaire
class FeaturesResponse(BaseModel):
    client_id: int
    features: Dict[str, Any]


# Pour envoyer une requête de demande de prédiction, le ID client doit être un int
class PredictionRequest(BaseModel):
    # client_id: int = Field(..., example=1234)
    client_id: int = Field(...)

    # Remplace le mot clef example dans Field() car c'est deprecated
    model_config = ConfigDict(json_schema_extra={"example": {"client_id": 258223}})


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
