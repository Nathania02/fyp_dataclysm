from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

class UserRole(str, enum.Enum):
    DATA_SCIENTIST = "data_scientist"
    CLINICIAN = "clinician"

class ModelType(str, enum.Enum):
    KMEANS = "kmeans"
    KMEANS_DTW = "kmeans_dtw"
    LCA = "lca"

class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(Enum(UserRole))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    runs = relationship("ModelRun", back_populates="user")
    notifications = relationship("Notification", back_populates="user")

class ModelRun(Base):
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_type = Column(Enum(ModelType))
    status = Column(Enum(RunStatus), default=RunStatus.PENDING)
    folder_path = Column(String)
    dataset_filename = Column(String)
    optimal_clusters = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    celery_task_id = Column(String, nullable=True)
    sent_to_clinician = Column(Boolean, default=False)
    feedback_added = Column(Boolean, default=False)
    
    user = relationship("User", back_populates="runs")
    notifications = relationship("Notification", back_populates="run")

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    run_id = Column(Integer, ForeignKey("model_runs.id"))
    message = Column(Text)
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="notifications")
    run = relationship("ModelRun", back_populates="notifications")