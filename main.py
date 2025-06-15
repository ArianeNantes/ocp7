from fastapi import FastAPI
import joblib
import os

# from src.api.routes import router as prediction_router
# from src.api.predictor import load_model, load_data_test, load_threshold

app = FastAPI(
    # title="API Prédiction Client",
    # description="API pour prédire un résultat à partir de l'ID client",
    # version="1.0.0",
)


# load_model()
# load_threshold()
# load_data_test()

# Ajouter les routes
# app.include_router(prediction_router)

# Chemin absolu vers le modèle (dans models/ car le dossier api/ est déployé à la racine)
model_path = os.path.join(os.path.dirname(__file__), "models", "best_model_lgbm.pkl")
model = joblib.load(model_path)


@app.get("/")
def root():
    return {"model_path": model_path, "model_class": model.__class__.__name__}
