from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.schemas import OrderInput, PredictionResponse
from deep_learning.src.offline_testing import OfflineTester   # <-- yangi yo'l

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "deep_learning" / "models"             # <-- yangi yo'l
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tester"] = OfflineTester(
        model_path=MODELS_DIR / "best_model.pth",
        preprocessor_path=MODELS_DIR / "preprocessor.joblib",
    )
    yield
    state.clear()

app = FastAPI(title="Order Return Prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "tester" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    tester = state["tester"]
    p = float(tester.predict_proba(order.model_dump())[0])
    pred = int(p > tester.threshold)

    return PredictionResponse(
        return_probability=round(p, 4),
        no_return_probability=round(1 - p, 4),
        prediction=pred,
        label="WILL BE RETURNED" if pred else "WILL NOT BE RETURNED",
    )