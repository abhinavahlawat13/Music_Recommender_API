from typing import List
from fastapi import APIRouter, FastAPI, HTTPException, Query

from app.schemas.recommendation import RecommendationResponse
from app.ml.recommender import recommender_engine

# Mini router
router = APIRouter()

@router.get("/", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    track_name: str = Query(..., description="The name of the track for which to get recommendations."),
    top_k: int = Query(5, ge=1, le=20, description="The number of top recommendations to return (default is 5).")
):
    try:
        results = recommender_engine.recommend(track_name, top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    if not results:
        raise HTTPException(status_code=404, detail="No recommendations found try another track.")

    
    return results