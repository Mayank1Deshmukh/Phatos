from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx

app = FastAPI()
HF_API_URL = "https://slashhash-phatos-space.hf.space"

class PredictRequest(BaseModel):
    image_b64: str

class ExplainRequest(BaseModel):
    image_b64: str
    emotion:   Optional[str] = None

class AnnotateRequest(BaseModel):
    painting_id:     int
    guessed_emotion: str

async def _post(client: httpx.AsyncClient, path: str, payload: dict):
    try:
        r = await client.post(f"{HF_API_URL}{path}", json=payload, timeout=60.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception:
        raise HTTPException(status_code=500, detail="Backend call failed")

async def _get(client: httpx.AsyncClient, path: str, params: dict):
    try:
        r = await client.get(f"{HF_API_URL}{path}", params=params, timeout=60.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception:
        raise HTTPException(status_code=500, detail="Backend call failed")

@app.post("/api/predict")
async def predict(req: PredictRequest):
    async with httpx.AsyncClient() as c:
        return await _post(c, "/api/predict", {"image_b64": req.image_b64})

@app.post("/api/explain")
async def explain(req: ExplainRequest):
    payload = {"image_b64": req.image_b64}
    if req.emotion:
        payload["emotion"] = req.emotion
    async with httpx.AsyncClient() as c:
        return await _post(c, "/api/explain", payload)

@app.post("/api/annotate", status_code=201)
async def post_annotate(req: AnnotateRequest):
    async with httpx.AsyncClient() as c:
        return await _post(c, "/api/annotate",
                           {"painting_id": req.painting_id,
                            "guessed_emotion": req.guessed_emotion})

@app.get("/api/annotate")
async def get_annotate(painting_id: int):
    async with httpx.AsyncClient() as c:
        return await _get(c, "/api/annotate", {"painting_id": painting_id})