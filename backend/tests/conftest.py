"""
Pytest configuration and fixtures for testing
"""
import os
import json
import pytest
import tempfile
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.storage import UserStorage, RunStorage, NotificationStorage

@pytest.fixture(scope="function")
def test_storage_files():
    """Create temporary storage files for testing"""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Store original paths
    original_users = settings.USERS_FILE
    original_runs = settings.RUNS_FILE
    original_notifications = settings.NOTIFICATIONS_FILE
    original_results = settings.RESULTS_DIR
    
    # Set test paths
    settings.USERS_FILE = os.path.join(temp_dir, "test_users.json")
    settings.RUNS_FILE = os.path.join(temp_dir, "test_runs.json")
    settings.NOTIFICATIONS_FILE = os.path.join(temp_dir, "test_notifications.json")
    settings.RESULTS_DIR = os.path.join(temp_dir, "results")
    
    # Initialize empty files
    with open(settings.USERS_FILE, 'w') as f:
        json.dump([], f)
    with open(settings.RUNS_FILE, 'w') as f:
        json.dump([], f)
    with open(settings.NOTIFICATIONS_FILE, 'w') as f:
        json.dump([], f)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)
    
    yield temp_dir
    
    # Restore original paths
    settings.USERS_FILE = original_users
    settings.RUNS_FILE = original_runs
    settings.NOTIFICATIONS_FILE = original_notifications
    settings.RESULTS_DIR = original_results
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def client(test_storage_files):
    """Create test client"""
    return TestClient(app)

@pytest.fixture
def test_user(test_storage_files):
    """Create a test user"""
    from app.auth import get_password_hash
    user_data = {
        'email': 'test@example.com',
        'hashed_password': get_password_hash('testpassword123'),
        'role': 'data_scientist'
    }
    return UserStorage.create(user_data)

@pytest.fixture
def test_clinician(test_storage_files):
    """Create a test clinician user"""
    from app.auth import get_password_hash
    user_data = {
        'email': 'clinician@example.com',
        'hashed_password': get_password_hash('clinicianpass123'),
        'role': 'clinician'
    }
    return UserStorage.create(user_data)

@pytest.fixture
def auth_token(client, test_user):
    """Get authentication token for test user"""
    response = client.post(
        "/api/auth/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    return response.json()["access_token"]

@pytest.fixture
def clinician_token(client, test_clinician):
    """Get authentication token for clinician user"""
    response = client.post(
        "/api/auth/login",
        json={
            "email": "clinician@example.com",
            "password": "clinicianpass123"
        }
    )
    return response.json()["access_token"]

@pytest.fixture
def test_run(test_storage_files, test_user):
    """Create a test run"""
    run_data = {
        'user_id': test_user['id'],
        'user_email': test_user['email'],
        'model_type': 'kmeans',
        'dataset_filename': 'test_dataset.duckdb',
        'dataset_name': 'test_table',
        'parameters_filename': 'test_params.yaml'
    }
    return RunStorage.create(run_data)

@pytest.fixture
def completed_run(test_storage_files, test_user):
    """Create a completed test run"""
    run_data = {
        'user_id': test_user['id'],
        'user_email': test_user['email'],
        'model_type': 'kmeans',
        'dataset_filename': 'test_dataset.duckdb',
        'dataset_name': 'test_table',
        'parameters_filename': 'test_params.yaml'
    }
    run = RunStorage.create(run_data)
    RunStorage.update(run['id'], {
        'status': 'completed',
        'optimal_clusters': 3,
        'folder_path': '/test/path',
        'completed_at': '2025-01-01T00:00:00'
    })
    return RunStorage.get_by_id(run['id'])

@pytest.fixture
def test_notification(test_storage_files, test_user, test_run):
    """Create a test notification"""
    notification_data = {
        'user_id': test_user['id'],
        'run_id': test_run['id'],
        'message': 'Test notification message'
    }
    return NotificationStorage.create(notification_data)
