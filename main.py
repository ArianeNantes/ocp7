from fastapi import FastAPI

# from src.api.routes import router as prediction_router
# from src.api.predictor import load_model, load_data_test, load_threshold

app = FastAPI(
    title="API Prédiction Client",
    description="API pour prédire un résultat à partir de l'ID client",
    version="1.0.0",
)

# Charger le modèle et les données au démarrage
"""@app.on_event("startup")
def startup_event():"""
"""
load_model()
load_threshold()
load_data_test()

# Ajouter les routes
app.include_router(prediction_router)
"""


@app.get("/")
def read_root():
    return "Hello Azure"
