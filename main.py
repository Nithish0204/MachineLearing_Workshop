from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine import SentimentEngine
from functools import lru_cache

app = FastAPI(title="Optimized Production AI API")
model_engine = SentimentEngine()

# 1. Simple In-Memory Cache (Least Recently Used)
# This prevents re-computing the same sentiment for common inputs like "Thank you!"
@lru_cache(maxsize=128)
def get_cached_prediction(text: str):
    return model_engine.predict(text)

class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)

class BatchAnalysisRequest(BaseModel):
    # This allows a single API call to process a list of 100 texts at once
    texts: list[str] = Field(..., min_size=1, max_size=50)

# 2. Optimized Prediction Endpoint (with Caching)
@app.post("/predict")
async def predict_sentiment(request: AnalysisRequest):
    # First, check the cache
    result = get_cached_prediction(request.text)
    return result

# 3. High-Throughput Batch Endpoint
@app.post("/predict_batch")
async def predict_batch(request: BatchAnalysisRequest):
    try:
        # Instead of a for-loop (slow), we use the model's built-in batching (fast)
        results = model_engine.predict_batch(request.texts)
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Batch inference failed.")

# Essential for Production: Load Balancers use this to see if the container is alive.
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model_engine.classifier is not None}

# To run: uvicorn main:app --reload