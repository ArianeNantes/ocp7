from fastapi import FastAPI
from routes import router as prediction_router
from predictor import load_model, load_data_test, load_threshold


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
