# src/routes/predict.py
from fastapi import APIRouter, HTTPException
from src.api.shema import (
    FeaturesResponse,
    PredictionRequest,
    PredictionResponse,
    ProbaResponse,
)
from src.api.predictor import (
    get_features_from_id,
    predict_from_features,
    predict_proba_from_features,
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
    # return{"client_id": client_id, "features": features}
    return FeaturesResponse(client_id=client_id, features=features)


# Route pour la requête POST qui renvoie la réponse complète, à savoir les données client, sa proba de défaut, le seuil et la target prédite
# à partir de l'ID
# Nota :
# Si le serveur ne connaissait pas les données ou si nous autorisions depuis des données externes/modifiées, nous devrions prévoir une autre route
# du type :
# @router.post("/predict_from_features", response_model=PredictionResponse)
# def predict_from_input(features: dict):
#    proba, prediction, threshold = predict_from_features(features)
#    return PredictionResponse(
#        client_id=None,  # ou un champ optionnel
#        features=features,
#        proba=proba,
#        prediction=prediction,
#        threshold=threshold,
#    )
# avec les schémas pydantic correspondant.
@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    features = get_features_from_id(request.client_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail="Impossible de faire la prédiction : ID client inconnu.",
        )
    proba, prediction, threshold = predict_from_features(features)

    return PredictionResponse(
        client_id=request.client_id,
        features=features,
        proba=proba,
        prediction=prediction,
        threshold=threshold,
    )


# Route pour la requête POST qui renvoie la réponse avec la probabilité de défaut mais pas la décision finale
@router.post("/predict_proba", response_model=ProbaResponse)
def predict_proba(request: PredictionRequest):
    features = get_features_from_id(request.client_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail="Impossible de faire la prédiction : ID client inconnu.",
        )
    prediction_proba = predict_proba_from_features(features)
    # return {"client_id": request.client_id, "prediction_proba": prediction_proba}
    return ProbaResponse(
        client_id=request.client_id,
        proba=prediction_proba,
    )
