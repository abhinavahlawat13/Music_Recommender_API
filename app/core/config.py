from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Music Recommender Engine"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "supersecret_recommendation_key_change_in_production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    class Config:
        case_sensitive = True


settings = Settings()