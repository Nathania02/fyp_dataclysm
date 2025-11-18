"""
Integration tests for end-to-end workflows
"""
import pytest
import os
from unittest.mock import patch

class TestCompleteWorkflow:
    """Test complete user workflows from start to finish"""
    
    def test_data_scientist_workflow(self, client, test_storage_files):
        """Test complete data scientist workflow"""
        # 1. Sign up
        signup_response = client.post(
            "/api/auth/signup",
            json={
                "email": "ds@example.com",
                "password": "password123",
                "role": "data_scientist"
            }
        )
        assert signup_response.status_code == 200
        
        # 2. Login
        login_response = client.post(
            "/api/auth/login",
            json={
                "email": "ds@example.com",
                "password": "password123"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # 3. Get user info
        me_response = client.get("/api/auth/me", headers=headers)
        assert me_response.status_code == 200
        assert me_response.json()["email"] == "ds@example.com"
        
        # 4. View runs (should be empty initially)
        runs_response = client.get("/api/runs", headers=headers)
        assert runs_response.status_code == 200
        assert len(runs_response.json()) == 0
        
        # 5. View notifications
        notifications_response = client.get("/api/notifications", headers=headers)
        assert notifications_response.status_code == 200
    
    @patch('app.routes.send_clinician_review_email')
    def test_clinician_feedback_workflow(self, mock_email, client, test_storage_files, test_user, test_clinician):
        """Test clinician review and feedback workflow"""
        mock_email.return_value = True
        
        # Login as data scientist
        ds_login = client.post(
            "/api/auth/login",
            json={"email": test_user['email'], "password": "testpassword123"}
        )
        ds_token = ds_login.json()["access_token"]
        ds_headers = {"Authorization": f"Bearer {ds_token}"}
        
        # Login as clinician
        clinician_login = client.post(
            "/api/auth/login",
            json={"email": test_clinician['email'], "password": "clinicianpass123"}
        )
        clinician_token = clinician_login.json()["access_token"]
        clinician_headers = {"Authorization": f"Bearer {clinician_token}"}
        
        # Create a completed run
        from app.storage import RunStorage
        run = RunStorage.create({
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        })
        RunStorage.update(run['id'], {
            'status': 'completed',
            'optimal_clusters': 3,
            'folder_path': os.path.join(test_storage_files, 'run_folder'),
            'completed_at': '2025-01-01T00:00:00'
        })
        run = RunStorage.get_by_id(run['id'])
        
        # Create folder for notes
        os.makedirs(run['folder_path'], exist_ok=True)
        
        # Data scientist sends to clinician
        send_response = client.post(
            f"/api/runs/{run['id']}/send-to-clinician",
            headers=ds_headers
        )
        assert send_response.status_code == 200
        
        # Clinician views runs (should see the sent run)
        clinician_runs = client.get("/api/runs", headers=clinician_headers)
        assert clinician_runs.status_code == 200
        assert len(clinician_runs.json()) == 1
        
        # Clinician adds feedback
        feedback_response = client.post(
            f"/api/runs/{run['id']}/feedback",
            headers=clinician_headers,
            json={"feedback": "Great analysis!"}
        )
        assert feedback_response.status_code == 200
        
        # Data scientist receives notification
        notifications = client.get("/api/notifications", headers=ds_headers)
        assert notifications.status_code == 200
        assert len(notifications.json()) >= 1
    
    def test_multiple_users_isolation(self, client, test_storage_files):
        """Test that users can only see their own data"""
        # Create two users
        client.post(
            "/api/auth/signup",
            json={"email": "user1@example.com", "password": "pass1", "role": "data_scientist"}
        )
        client.post(
            "/api/auth/signup",
            json={"email": "user2@example.com", "password": "pass2", "role": "data_scientist"}
        )
        
        # Login both users
        user1_token = client.post(
            "/api/auth/login",
            json={"email": "user1@example.com", "password": "pass1"}
        ).json()["access_token"]
        
        user2_token = client.post(
            "/api/auth/login",
            json={"email": "user2@example.com", "password": "pass2"}
        ).json()["access_token"]
        
        # Create runs for user1
        from app.storage import RunStorage, UserStorage
        user1 = UserStorage.get_by_email("user1@example.com")
        RunStorage.create({
            'user_id': user1['id'],
            'user_email': user1['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        })
        
        # User2 gets notifications (should be empty)
        user2_notifications = client.get(
            "/api/notifications",
            headers={"Authorization": f"Bearer {user2_token}"}
        )
        assert len(user2_notifications.json()) == 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_json_payload(self, client, test_storage_files):
        """Test handling of invalid JSON"""
        response = client.post(
            "/api/auth/signup",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client, test_storage_files):
        """Test handling of missing required fields"""
        response = client.post(
            "/api/auth/signup",
            json={"email": "test@example.com"}  # Missing password and role
        )
        assert response.status_code == 422
    
    def test_nonexistent_run_id(self, client, test_storage_files, auth_token):
        """Test accessing non-existent run"""
        response = client.get(
            "/api/runs/99999",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        # Should return 404 for non-existent run
        assert response.status_code == 404
    
    def test_nonexistent_notification_id(self, client, test_storage_files, auth_token):
        """Test marking non-existent notification as read"""
        response = client.put(
            "/api/notifications/99999/read",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404
    
    def test_empty_email(self, client, test_storage_files):
        """Test signup with empty email"""
        response = client.post(
            "/api/auth/signup",
            json={"email": "", "password": "pass123", "role": "data_scientist"}
        )
        assert response.status_code == 422
    
    def test_empty_password(self, client, test_storage_files):
        """Test login with empty password"""
        response = client.post(
            "/api/auth/login",
            json={"email": "test@example.com", "password": ""}
        )
        assert response.status_code in [401, 422]
    
    def test_very_long_note(self, client, test_storage_files, auth_token, completed_run):
        """Test adding very long note"""
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        long_note = "A" * 10000  # 10k characters
        response = client.post(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"note": long_note}
        )
        assert response.status_code == 200
    
    def test_special_characters_in_feedback(self, client, test_storage_files, clinician_token, completed_run):
        """Test feedback with special characters"""
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        special_feedback = "Test with special chars: <>&\"'\\n\\t"
        response = client.post(
            f"/api/runs/{completed_run['id']}/feedback",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"feedback": special_feedback}
        )
        assert response.status_code == 200
    
    def test_concurrent_run_updates(self, test_storage_files, test_user):
        """Test concurrent updates to the same run"""
        from app.storage import RunStorage
        
        run = RunStorage.create({
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        })
        
        # Simulate concurrent updates
        RunStorage.update(run['id'], {'status': 'running'})
        RunStorage.update(run['id'], {'optimal_clusters': 3})
        
        updated_run = RunStorage.get_by_id(run['id'])
        assert updated_run['status'] == 'running'
        assert updated_run['optimal_clusters'] == 3


class TestSecurityScenarios:
    """Test security-related scenarios"""
    
    def test_expired_token(self, client, test_storage_files):
        """Test using expired token"""
        from datetime import timedelta
        from app.auth import create_access_token
        
        # Create token that expires immediately
        token = create_access_token(
            {"sub": "test@example.com"},
            expires_delta=timedelta(seconds=-1)
        )
        
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 401
    
    def test_malformed_token(self, client, test_storage_files):
        """Test using malformed token"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer malformed_token"}
        )
        assert response.status_code == 401
    
    def test_token_without_bearer_prefix(self, client, test_storage_files, auth_token):
        """Test using token without Bearer prefix"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": auth_token}
        )
        assert response.status_code == 401
    
    def test_sql_injection_in_email(self, client, test_storage_files):
        """Test SQL injection attempt in email field"""
        response = client.post(
            "/api/auth/signup",
            json={
                "email": "test@example.com' OR '1'='1",
                "password": "password123",
                "role": "data_scientist"
            }
        )
        # Should either fail validation or create user safely
        assert response.status_code in [200, 422]
    
    def test_xss_in_note(self, client, test_storage_files, auth_token, completed_run):
        """Test XSS attempt in note content"""
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        xss_note = "<script>alert('XSS')</script>"
        response = client.post(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"note": xss_note}
        )
        # Should accept but properly escape when displaying
        assert response.status_code == 200


class TestDataConsistency:
    """Test data consistency and integrity"""
    
    def test_user_id_increment(self, test_storage_files):
        """Test that user IDs increment correctly"""
        from app.storage import UserStorage
        from app.auth import get_password_hash
        
        user1 = UserStorage.create({
            'email': 'user1@example.com',
            'hashed_password': get_password_hash('pass1'),
            'role': 'data_scientist'
        })
        
        user2 = UserStorage.create({
            'email': 'user2@example.com',
            'hashed_password': get_password_hash('pass2'),
            'role': 'clinician'
        })
        
        assert user2['id'] == user1['id'] + 1
    
    def test_notification_creation_integrity(self, test_storage_files, test_user, test_run):
        """Test notification creation maintains data integrity"""
        from app.storage import NotificationStorage
        
        notification = NotificationStorage.create({
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Test message'
        })
        
        retrieved = NotificationStorage.get_by_id(notification['id'])
        
        assert retrieved['user_id'] == test_user['id']
        assert retrieved['run_id'] == test_run['id']
        assert retrieved['message'] == notification['message']
    
    def test_run_status_transitions(self, test_storage_files, test_run):
        """Test valid run status transitions"""
        from app.storage import RunStorage
        
        # pending -> running
        RunStorage.update(test_run['id'], {'status': 'running'})
        run = RunStorage.get_by_id(test_run['id'])
        assert run['status'] == 'running'
        
        # running -> completed
        RunStorage.update(test_run['id'], {'status': 'completed'})
        run = RunStorage.get_by_id(test_run['id'])
        assert run['status'] == 'completed'
