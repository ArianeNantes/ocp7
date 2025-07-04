from fastapi import APIRouter, HTTPException
from .shema import (
    FeaturesResponse,
    PredictionRequest,
    PredictionResponse,
    ProbaResponse,
)
from .predictor import (
    get_features_from_id,
    predict_from_features,
)

router = APIRouter()


# Route pour la Requête GET qui à partir d'un ID client retourne ses données,
# que l'on peut utiliser ensuite pour prédire
@router.get("/features/{client_id}", response_model=FeaturesResponse)
def read_features(client_id: int):
    features = get_features_from_id(client_id)
    if features is None:
        raise HTTPException(
            status_code=404, detail="Désolé, cet ID client n'est pas répertorié."
        )
    return FeaturesResponse(client_id=client_id, features=features)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    client_id = request.client_id
    features = request.features
    if features is None:
        raise HTTPException(
            status_code=404,
            detail="Impossible de faire la prédiction - Données client inconnues.",
        )
    proba, prediction, threshold = predict_from_features(features)

    return PredictionResponse(
        client_id=client_id,
        features=features,
        proba=proba,
        prediction=prediction,
        threshold=threshold,
    )
