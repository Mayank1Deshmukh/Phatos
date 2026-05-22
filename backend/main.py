from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx

app = FastAPI()
HF_API_URL = "https://slashhash-phatos-space.hf.space"

class PredictRequest(BaseModel):
    image_url: str

class ExplainRequest(BaseModel):
    image_url: str
    emotion: str | None = None

class AnnotateRequest(BaseModel):
    painting_id: int
    guessed_emotion: str

@app.post("/api/predict")
async def predict(req: PredictRequest):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{HF_API_URL}/api/predict",
                json={"image_url": req.image_url},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")

@app.post("/api/explain")
async def explain(req: ExplainRequest):
    payload = {"image_url": req.image_url}
    if req.emotion:
        payload["emotion"] = req.emotion
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{HF_API_URL}/api/explain",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")

@app.get("/api/search")
async def search(emotion: str, limit: int = 10):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{HF_API_URL}/api/search",
                params={"emotion": emotion, "limit": limit},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")

@app.get("/api/embeddings")
async def embeddings(movement: str | None = None, century: int | None = None):
    params = {}
    if movement:
        params["movement"] = movement
    if century is not None:
        params["century"] = century
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{HF_API_URL}/api/embeddings",
                params=params,
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")

@app.post("/api/annotate")
async def annotate(req: AnnotateRequest):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{HF_API_URL}/api/annotate",
                json={"painting_id": req.painting_id, "guessed_emotion": req.guessed_emotion},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")

@app.get("/api/annotate")
async def get_annotate_stats(painting_id: int):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(
                f"{HF_API_URL}/api/annotate",
                params={"painting_id": painting_id},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Backend call failed")