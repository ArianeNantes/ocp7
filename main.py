import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from src.api.routes import router as prediction_router

from src.api.routes import router as prediction_router
from src.api.predictor import load_model, load_data_test, load_threshold

app = FastAPI()


model = load_model()
threshold = load_threshold()
df = load_data_test()

# Ajouter les routes
app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "model_class": model.__class__.__name__,
        "threshold": threshold,
        "shape df": df.shape,
    }
