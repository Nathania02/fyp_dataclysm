"""
Unit tests for schemas/models validation
"""
import pytest
from pydantic import ValidationError
from app.schemas import (
    UserCreate, UserLogin, UserResponse, Token,
    ModelRunCreate, ModelRunResponse,
    NotificationResponse, FeedbackRequest, NoteRequest,
    UserRole, ModelType, RunStatus
)

class TestEnums:
    """Test enum validations"""
    
    def test_user_role_values(self):
        """Test UserRole enum values"""
        assert UserRole.DATA_SCIENTIST.value == "data_scientist"
        assert UserRole.CLINICIAN.value == "clinician"
    
    def test_model_type_values(self):
        """Test ModelType enum values"""
        assert ModelType.KMEANS.value == "kmeans"
        assert ModelType.KMEANS_DTW.value == "kmeans_dtw"
        assert ModelType.LCA.value == "lca"
    
    def test_run_status_values(self):
        """Test RunStatus enum values"""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"


class TestUserSchemas:
    """Test user-related schemas"""
    
    def test_user_create_valid(self):
        """Test valid UserCreate schema"""
        user = UserCreate(
            email="test@example.com",
            password="password123",
            role=UserRole.DATA_SCIENTIST
        )
        
        assert user.email == "test@example.com"
        assert user.password == "password123"
        assert user.role == UserRole.DATA_SCIENTIST
    
    def test_user_create_invalid_email(self):
        """Test UserCreate with invalid email"""
        with pytest.raises(ValidationError):
            UserCreate(
                email="invalid-email",
                password="password123",
                role=UserRole.DATA_SCIENTIST
            )
    
    def test_user_create_missing_fields(self):
        """Test UserCreate with missing fields"""
        with pytest.raises(ValidationError):
            UserCreate(email="test@example.com")
    
    def test_user_login_valid(self):
        """Test valid UserLogin schema"""
        login = UserLogin(
            email="test@example.com",
            password="password123"
        )
        
        assert login.email == "test@example.com"
        assert login.password == "password123"
    
    def test_token_schema(self):
        """Test Token schema"""
        token = Token(
            access_token="test_token_string",
            token_type="bearer"
        )
        
        assert token.access_token == "test_token_string"
        assert token.token_type == "bearer"
    
    def test_user_response_schema(self):
        """Test UserResponse schema"""
        user = UserResponse(
            id=1,
            email="test@example.com",
            role="data_scientist",
            created_at="2025-01-01T00:00:00"
        )
        
        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.role == "data_scientist"


class TestModelRunSchemas:
    """Test model run-related schemas"""
    
    def test_model_run_create_valid(self):
        """Test valid ModelRunCreate schema"""
        run = ModelRunCreate(
            model_type=ModelType.KMEANS,
            dataset_name="test_dataset",
            dataset_details="testing_dataset"
        )
        
        assert run.model_type == ModelType.KMEANS
        assert run.dataset_name == "test_dataset"
        assert run.dataset_details == "testing_dataset"
    
    def test_model_run_create_with_string_enum(self):
        """Test ModelRunCreate with string enum value"""
        run = ModelRunCreate(
            model_type="kmeans",
            dataset_name="test_dataset",
            dataset_details="testing_dataset"
        )
        
        assert run.model_type == ModelType.KMEANS
    
    def test_model_run_create_invalid_model_type(self):
        """Test ModelRunCreate with invalid model type"""
        with pytest.raises(ValidationError):
            ModelRunCreate(
                model_type="invalid_model",
                dataset_name="test_dataset",
                dataset_details="testing_dataset"
            )
    
    def test_model_run_response_complete(self):
        """Test complete ModelRunResponse schema"""
        run = ModelRunResponse(
            id=1,
            user_id=1,
            model_type="kmeans",
            status="completed",
            folder_path="/path/to/folder",
            dataset_filename="dataset.duckdb",
            optimal_clusters=3,
            created_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T01:00:00",
            sent_to_clinician=True,
            feedback_added=False
        )
        
        assert run.id == 1
        assert run.model_type == "kmeans"
        assert run.optimal_clusters == 3
    
    def test_model_run_response_optional_fields(self):
        """Test ModelRunResponse with optional fields as None"""
        run = ModelRunResponse(
            id=1,
            user_id=1,
            model_type="kmeans",
            status="pending",
            folder_path=None,
            dataset_filename=None,
            optimal_clusters=None,
            created_at="2025-01-01T00:00:00",
            completed_at=None,
            sent_to_clinician=False,
            feedback_added=False
        )
        
        assert run.folder_path is None
        assert run.optimal_clusters is None
        assert run.completed_at is None


class TestNotificationSchemas:
    """Test notification-related schemas"""
    
    def test_notification_response_valid(self):
        """Test valid NotificationResponse schema"""
        notification = NotificationResponse(
            id=1,
            user_id=1,
            run_id=1,
            message="Test notification",
            is_read=False,
            created_at="2025-01-01T00:00:00"
        )
        
        assert notification.id == 1
        assert notification.message == "Test notification"
        assert notification.is_read is False
    
    def test_notification_response_missing_fields(self):
        """Test NotificationResponse with missing required fields"""
        with pytest.raises(ValidationError):
            NotificationResponse(
                id=1,
                user_id=1,
                message="Test notification"
            )


class TestRequestSchemas:
    """Test request-related schemas"""
    
    def test_feedback_request_valid(self):
        """Test valid FeedbackRequest schema"""
        feedback = FeedbackRequest(feedback="This is clinical feedback")
        
        assert feedback.feedback == "This is clinical feedback"
    
    def test_feedback_request_empty(self):
        """Test FeedbackRequest with empty string"""
        feedback = FeedbackRequest(feedback="")
        
        assert feedback.feedback == ""
    
    def test_note_request_valid(self):
        """Test valid NoteRequest schema"""
        note = NoteRequest(note="This is a note")
        
        assert note.note == "This is a note"
    
    def test_note_request_missing_field(self):
        """Test NoteRequest with missing field"""
        with pytest.raises(ValidationError):
            NoteRequest()


class TestSchemaSerializaton:
    """Test schema serialization"""
    
    def test_user_response_dict(self):
        """Test UserResponse conversion to dict"""
        user = UserResponse(
            id=1,
            email="test@example.com",
            role="data_scientist",
            created_at="2025-01-01T00:00:00"
        )
        
        user_dict = user.model_dump()
        
        assert user_dict['id'] == 1
        assert user_dict['email'] == "test@example.com"
    
    def test_model_run_response_dict(self):
        """Test ModelRunResponse conversion to dict"""
        run = ModelRunResponse(
            id=1,
            user_id=1,
            model_type="kmeans",
            status="completed",
            folder_path="/path",
            dataset_filename="data.duckdb",
            optimal_clusters=3,
            created_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T01:00:00",
            sent_to_clinician=False,
            feedback_added=False
        )
        
        run_dict = run.model_dump()
        
        assert run_dict['id'] == 1
        assert run_dict['optimal_clusters'] == 3
