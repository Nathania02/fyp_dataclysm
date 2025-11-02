from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # JWT Configuration
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200
    
    # Redis Configuration
    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Storage Configuration
    RESULTS_DIR: str = "model_runs_results"
    USERS_FILE: str = "users.json"
    RUNS_FILE: str = "runs.json"
    NOTIFICATIONS_FILE: str = "notifications.json"
    
    class Config:
        env_file = ".env",
        extra = "ignore"

settings = Settings()