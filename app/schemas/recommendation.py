from pydantic import BaseModel
from typing import List, Optional

class SongSummary(BaseModel):
    track_name: str
    artist_name: str
    genre: Optional[str] = "Unknown"
    similarity_score: Optional[float] = None
    energy: Optional[float] = None
    valence: Optional[float] = None


# 2. Main API response ka wrapper schema
class RecommendationResponse(BaseModel):
    query_song: SongSummary
    recommendations: List[SongSummary]