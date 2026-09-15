"""Local demo: independent VM rules and experimental Alibaba predictions."""
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from backend.app.services.recommendation_service import recommend_resources
from ML.src.predict import predict_resources

app = FastAPI(title="AI Cloud Optimizer", version="0.2.0")
PROJECT = Path(__file__).resolve().parents[2]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instance_num: int = Field(strict=True, gt=0, le=1_000_000)
    plan_cpu: float = Field(gt=0)
    plan_mem: float = Field(gt=0)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static/index.html")


@app.get("/health")
def health():
    return {"status": "healthy", "models_available": all(
        (PROJECT / "ML/models/reference_v1" / f"{r}_reference.joblib").is_file()
        for r in ("cpu", "mem")
    )}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    return recommend_resources(request)


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        return predict_resources(**request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(503, "Modèles absents du serveur.") from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Model prediction failed")
        raise HTTPException(503, "Prédiction indisponible. Vérifier les modèles et leur environnement Python.") from exc


@app.post("/terraform/variables")
def terraform_variables(request: RecommendationRequest):
    recommendation = recommend_resources(request)
    variables = {
        "vcpu": recommendation.recommended_vcpu,
        "memory_mb": recommendation.recommended_ram_gb * 1024,
        "disk_gb": recommendation.recommended_disk_gb,
        "vm_count": recommendation.vm_count,
    }
    return Response(json.dumps(variables, indent=2) + "\n", media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="sizing.auto.tfvars.json"'})
