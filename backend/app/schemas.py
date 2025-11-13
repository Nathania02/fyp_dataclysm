from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, List
from enum import Enum

class UserRole(str, Enum):
    DATA_SCIENTIST = "data_scientist"
    CLINICIAN = "clinician"

class ModelType(str, Enum):
    KMEANS = "kmeans"
    KMEANS_DTW = "kmeans_dtw"
    LCA = "lca"
    GBTM = "gbtm"

class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class UserCreate(BaseModel):
    email: str
    password: str
    role: UserRole

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    created_at: str

# note: can possibly remove
class ModelRunCreate(BaseModel):
    model_type: ModelType
    dataset_name: str
    dataset_details: str

class ModelRunResponse(BaseModel):
    id: int
    user_id: int
    model_type: str
    status: str
    folder_path: Optional[str]
    dataset_filename: Optional[str]
    optimal_clusters: Optional[int]
    created_at: str
    completed_at: Optional[str]
    sent_to_clinician: bool
    feedback_added: bool

class NotificationResponse(BaseModel):
    id: int
    user_id: int
    run_id: int
    message: str
    is_read: bool
    created_at: str

class FeedbackRequest(BaseModel):
    feedback: str

class NoteRequest(BaseModel):
    note: str