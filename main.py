from app.api import recommendation

from fastapi import FastAPI
from app.core.config import settings
from contextlib import asynccontextmanager
from typing import List

# importing the recommender engine
from app.ml.recommender import recommender_engine
# --- Lifespan: it keeps the server alive and loads the dataset and vectors into memory on startup, and cleans up on shutdown.

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[STARTUP] loading dataset and vectors...")
    recommender_engine.load_and_prepare(limit=50000)
    print("[STARTUP] Music Engine Ready!\n")
    yield
    print("\n[SHUTDOWN] closing resources...")


# FASTAPI initialization
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.include_router(
    recommendation.router,
    prefix="/recommendations",
    tags=["Recommendations"]
)



# system health check route
@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME
    }

# Main home page route
@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Music Recommender Engine is online."
    }