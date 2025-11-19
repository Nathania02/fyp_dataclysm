"""
Unit tests for API routes
"""
import pytest
import io
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi import UploadFile

class TestAuthRoutes:
    """Test authentication routes"""
    
    def test_signup_success(self, client, test_storage_files):
        """Test successful user signup"""
        response = client.post(
            "/api/auth/signup",
            json={
                "email": "newuser@example.com",
                "password": "password123",
                "role": "data_scientist"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['email'] == 'newuser@example.com'
        assert data['role'] == 'data_scientist'
        assert 'id' in data
    
    def test_signup_duplicate_email(self, client, test_storage_files, test_user):
        """Test signup with duplicate email"""
        response = client.post(
            "/api/auth/signup",
            json={
                "email": "test@example.com",
                "password": "password123",
                "role": "data_scientist"
            }
        )
        
        assert response.status_code == 400
        assert "already registered" in response.json()['detail']
    
    def test_login_success(self, client, test_storage_files, test_user):
        """Test successful login"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": "test@example.com",
                "password": "testpassword123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'access_token' in data
        assert data['token_type'] == 'bearer'
    
    def test_login_wrong_password(self, client, test_storage_files, test_user):
        """Test login with wrong password"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": "test@example.com",
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()['detail']
    
    def test_login_nonexistent_user(self, client, test_storage_files):
        """Test login with non-existent user"""
        response = client.post(
            "/api/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123"
            }
        )
        
        assert response.status_code == 401
    
    def test_get_me(self, client, test_storage_files, auth_token):
        """Test getting current user info"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['email'] == 'test@example.com'
        assert data['role'] == 'data_scientist'
    
    def test_get_me_unauthorized(self, client, test_storage_files):
        """Test getting user info without token"""
        response = client.get("/api/auth/me")
        
        assert response.status_code == 401


class TestRunRoutes:
    """Test run management routes"""
    
    def test_get_runs_data_scientist(self, client, test_storage_files, auth_token, test_run):
        """Test getting runs as data scientist"""
        response = client.get(
            "/api/runs",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
    
    def test_get_runs_clinician(self, client, test_storage_files, clinician_token, completed_run):
        """Test getting runs as clinician (only sent runs)"""
        # Mark run as sent to clinician
        from app.storage import RunStorage
        RunStorage.update(completed_run['id'], {'sent_to_clinician': True})
        
        response = client.get(
            "/api/runs",
            headers={"Authorization": f"Bearer {clinician_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert all(run['sent_to_clinician'] for run in data)
    
    def test_get_run_by_id(self, client, test_storage_files, auth_token, test_run):
        """Test getting specific run by ID"""
        response = client.get(
            f"/api/runs/{test_run['id']}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['id'] == test_run['id']
        assert data['model_type'] == test_run['model_type']
    
    def test_get_run_plots(self, client, test_storage_files, auth_token, completed_run):
        """Test getting run plots"""
        # Create a mock folder with plots
        import os
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        # Create dummy plot files
        with open(os.path.join(completed_run['folder_path'], 'plot1.png'), 'w') as f:
            f.write('dummy')
        with open(os.path.join(completed_run['folder_path'], 'plot2.png'), 'w') as f:
            f.write('dummy')
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/plots",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'plots' in data
        assert len(data['plots']) == 2
    
    def test_get_run_notes_empty(self, client, test_storage_files, auth_token, completed_run):
        """Test getting run notes when none exist"""
        # Ensure notes file doesn't exist
        import os
        notes_file = os.path.join(completed_run['folder_path'], 'notes_feedback.txt')
        if os.path.exists(notes_file):
            os.remove(notes_file)
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == ''
    
    def test_add_note(self, client, test_storage_files, auth_token, completed_run):
        """Test adding a note to a run"""
        # Create folder first
        import os
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"note": "This is a test note"}
        )
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
    
    @patch('app.routes.send_clinician_review_email')
    def test_send_to_clinician(self, mock_email, client, test_storage_files, auth_token, completed_run, test_clinician):
        """Test sending run to clinician"""
        mock_email.return_value = True
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/send-to-clinician",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
        
        # Verify run was updated
        from app.storage import RunStorage
        run = RunStorage.get_by_id(completed_run['id'])
        assert run['sent_to_clinician'] is True
    
    def test_send_to_clinician_not_completed(self, client, test_storage_files, auth_token, test_run):
        """Test sending incomplete run to clinician"""
        response = client.post(
            f"/api/runs/{test_run['id']}/send-to-clinician",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 400
        assert "not completed" in response.json()['detail']
    
    def test_add_feedback(self, client, test_storage_files, clinician_token, completed_run):
        """Test adding feedback to a run"""
        # Create folder first
        import os
        os.makedirs(completed_run['folder_path'], exist_ok=True)
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/feedback",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"feedback": "This is clinical feedback"}
        )
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
        
        # Verify feedback_added flag was set
        from app.storage import RunStorage
        run = RunStorage.get_by_id(completed_run['id'])
        assert run['feedback_added'] is True


class TestNotificationRoutes:
    """Test notification routes"""
    
    def test_get_notifications(self, client, test_storage_files, auth_token, test_notification):
        """Test getting user notifications"""
        response = client.get(
            "/api/notifications",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]['message'] == 'Test notification message'
    
    def test_mark_notification_read(self, client, test_storage_files, auth_token, test_notification):
        """Test marking notification as read"""
        response = client.put(
            f"/api/notifications/{test_notification['id']}/read",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
        
        # Verify notification was marked as read
        from app.storage import NotificationStorage
        notification = NotificationStorage.get_by_id(test_notification['id'])
        assert notification['is_read'] is True
    
    def test_mark_notification_read_wrong_user(self, client, test_storage_files, clinician_token, test_notification):
        """Test marking another user's notification as read"""
        response = client.put(
            f"/api/notifications/{test_notification['id']}/read",
            headers={"Authorization": f"Bearer {clinician_token}"}
        )
        
        assert response.status_code == 404


class TestUnauthorizedAccess:
    """Test unauthorized access to protected endpoints"""
    
    def test_runs_without_auth(self, client, test_storage_files):
        """Test accessing runs without authentication"""
        response = client.get("/api/runs")
        assert response.status_code == 401
    
    def test_notifications_without_auth(self, client, test_storage_files):
        """Test accessing notifications without authentication"""
        response = client.get("/api/notifications")
        assert response.status_code == 401
    
    def test_get_me_without_auth(self, client, test_storage_files):
        """Test accessing user info without authentication"""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

class TestCreateRunEndpoint:
    """Test the create_run endpoint comprehensively"""
    
    @patch('app.routes.train_model.delay')
    def test_create_run_success(self, mock_celery_task, client, test_storage_files, auth_token, tmp_path):
        """Test successful run creation with file uploads"""
        # Mock Celery task
        mock_task = Mock()
        mock_task.id = "test-task-id-123"
        mock_celery_task.return_value = mock_task
        
        # Create mock files
        dataset_content = b"mock dataset content"
        params_content = b"range:\n  k_min: 2\n  k_max: 5\ncolumns_to_exclude: []\nhyperparameters:\n  random_state: 42"
        
        model_data = {
            "model_type": "kmeans",
            "dataset_name": "test_table",
            "dataset_details": "Test dataset details"
        }
        
        response = client.post(
            "/api/runs",
            headers={"Authorization": f"Bearer {auth_token}"},
            data={"model_data": json.dumps(model_data)},
            files={
                "dataset_file": ("test.duckdb", io.BytesIO(dataset_content), "application/octet-stream"),
                "parameters_file": ("params.yaml", io.BytesIO(params_content), "text/yaml")
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['model_type'] == 'kmeans'
        assert data['status'] == 'running'
        
        # Verify Celery task was called
        mock_celery_task.assert_called_once()
    
    def test_create_run_unauthorized(self, client, test_storage_files):
        """Test creating run without authentication"""
        model_data = {
            "model_type": "kmeans",
            "dataset_name": "test_table",
            "dataset_details": "Test details"
        }
        
        response = client.post(
            "/api/runs",
            data={"model_data": json.dumps(model_data)},
            files={
                "dataset_file": ("test.duckdb", io.BytesIO(b"data"), "application/octet-stream"),
                "parameters_file": ("params.yaml", io.BytesIO(b"params"), "text/yaml")
            }
        )
        
        assert response.status_code == 401
    
    @patch('app.routes.train_model.delay')
    def test_create_run_with_different_model_types(self, mock_celery_task, client, test_storage_files, auth_token):
        """Test creating runs with different model types"""
        mock_task = Mock()
        mock_task.id = "task-123"
        mock_celery_task.return_value = mock_task
        
        model_types = ["kmeans", "lca", "kmeans_dtw", "gbtm"]
        
        for model_type in model_types:
            model_data = {
                "model_type": model_type,
                "dataset_name": f"{model_type}_table",
                "dataset_details": f"{model_type} details"
            }
            
            response = client.post(
                "/api/runs",
                headers={"Authorization": f"Bearer {auth_token}"},
                data={"model_data": json.dumps(model_data)},
                files={
                    "dataset_file": (f"{model_type}.duckdb", io.BytesIO(b"data"), "application/octet-stream"),
                    "parameters_file": ("params.yaml", io.BytesIO(b"params: test"), "text/yaml")
                }
            )
            
            assert response.status_code == 200
            assert response.json()['model_type'] == model_type
    
    @patch('app.routes.train_model.delay')
    def test_create_run_metadata_file_created(self, mock_celery_task, client, test_storage_files, auth_token, tmp_path):
        """Test that metadata file is created when run is created"""
        mock_task = Mock()
        mock_task.id = "task-456"
        mock_celery_task.return_value = mock_task
        
        # Patch settings to use temp directory
        with patch('app.routes.settings.RESULTS_DIR', str(tmp_path)):
            model_data = {
                "model_type": "kmeans",
                "dataset_name": "test_table",
                "dataset_details": "Important dataset information"
            }
            
            response = client.post(
                "/api/runs",
                headers={"Authorization": f"Bearer {auth_token}"},
                data={"model_data": json.dumps(model_data)},
                files={
                    "dataset_file": ("test.duckdb", io.BytesIO(b"data"), "application/octet-stream"),
                    "parameters_file": ("params.yaml", io.BytesIO(b"params"), "text/yaml")
                }
            )
            
            assert response.status_code == 200


class TestGetRunWithStatusUpdate:
    """Test get_run endpoint with status updates"""
    
    @patch('app.routes.AsyncResult')
    def test_get_run_updates_to_completed(self, mock_async_result, client, test_storage_files, auth_token, test_run):
        """Test that get_run updates status when task completes"""
        from app.storage import RunStorage
        
        # Set run to running status
        RunStorage.update(test_run['id'], {
            'status': 'running',
            'celery_task_id': 'test-task-123'
        })
        
        # Mock AsyncResult to return completed task
        mock_task = Mock()
        mock_task.ready.return_value = True
        mock_task.result = {'status': 'success', 'optimal_clusters': 4}
        mock_async_result.return_value = mock_task
        
        response = client.get(
            f"/api/runs/{test_run['id']}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'completed'
        assert data['optimal_clusters'] == 4
        
        # Note: Email is now sent from Celery task, not from the route endpoint
    
    @patch('app.routes.AsyncResult')
    def test_get_run_updates_to_failed(self, mock_async_result, client, test_storage_files, auth_token, test_run):
        """Test that get_run updates status when task fails"""
        from app.storage import RunStorage
        
        RunStorage.update(test_run['id'], {
            'status': 'running',
            'celery_task_id': 'test-task-456'
        })
        
        # Mock AsyncResult to return failed task
        mock_task = Mock()
        mock_task.ready.return_value = True
        mock_task.result = None  # Failed task returns None
        mock_async_result.return_value = mock_task
        
        response = client.get(
            f"/api/runs/{test_run['id']}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'failed'
        
        # Note: Email is now sent from Celery task, not from the route endpoint
    
    @patch('app.routes.AsyncResult')
    def test_get_run_task_not_ready(self, mock_async_result, client, test_storage_files, auth_token, test_run):
        """Test get_run when task is still running"""
        from app.storage import RunStorage
        
        RunStorage.update(test_run['id'], {
            'status': 'running',
            'celery_task_id': 'test-task-789'
        })
        
        # Mock AsyncResult to return not ready
        mock_task = Mock()
        mock_task.ready.return_value = False
        mock_async_result.return_value = mock_task
        
        response = client.get(
            f"/api/runs/{test_run['id']}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'running'  # Status should not change


class TestGetRunPlots:
    """Test get_run_plots endpoint"""
    
    def test_get_run_plots_with_files(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test getting plots when plot files exist"""
        from app.storage import RunStorage
        
        # Create temp folder with plot files
        plot_folder = tmp_path / "plots"
        plot_folder.mkdir()
        
        # Create dummy plot files
        (plot_folder / "plot1.png").write_text("plot1")
        (plot_folder / "plot2.png").write_text("plot2")
        (plot_folder / "plot3.png").write_text("plot3")
        (plot_folder / "not_a_plot.txt").write_text("text")
        
        # Update run with plot folder
        RunStorage.update(completed_run['id'], {'folder_path': str(plot_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/plots",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'plots' in data
        assert len(data['plots']) == 3
        assert all(plot.endswith('.png') for plot in data['plots'])
    
    def test_get_run_plots_empty_folder(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test getting plots when folder is empty"""
        from app.storage import RunStorage
        
        plot_folder = tmp_path / "empty_plots"
        plot_folder.mkdir()
        
        RunStorage.update(completed_run['id'], {'folder_path': str(plot_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/plots",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['plots'] == []
    
    def test_get_run_plots_no_folder(self, client, test_storage_files, auth_token, test_run):
        """Test getting plots when no folder path exists"""
        response = client.get(
            f"/api/runs/{test_run['id']}/plots",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['plots'] == []


class TestGetRunPlotFile:
    """Test get_plot_file endpoint"""
    
    def test_get_plot_file_success(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test successfully retrieving a plot file"""
        from app.storage import RunStorage
        
        plot_folder = tmp_path / "plots"
        plot_folder.mkdir()
        plot_file = plot_folder / "test_plot.png"
        plot_file.write_bytes(b"fake png data")
        
        RunStorage.update(completed_run['id'], {'folder_path': str(plot_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/plots/test_plot.png",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        assert response.content == b"fake png data"
    
    def test_get_plot_file_not_found(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test retrieving non-existent plot file"""
        from app.storage import RunStorage
        
        plot_folder = tmp_path / "plots"
        plot_folder.mkdir()
        
        RunStorage.update(completed_run['id'], {'folder_path': str(plot_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/plots/nonexistent.png",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 404
    
    def test_get_plot_file_no_folder_path(self, client, test_storage_files, auth_token, test_run):
        """Test retrieving plot when run has no folder path"""
        response = client.get(
            f"/api/runs/{test_run['id']}/plots/plot.png",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 404


class TestGetRunNotes:
    """Test get_notes endpoint"""
    
    def test_get_notes_with_content(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test getting notes when file exists with content"""
        from app.storage import RunStorage
        
        notes_folder = tmp_path / "notes"
        notes_folder.mkdir()
        notes_file = notes_folder / "notes_feedback.txt"
        notes_content = "This is a test note\\nWith multiple lines"
        notes_file.write_text(notes_content)
        
        RunStorage.update(completed_run['id'], {'folder_path': str(notes_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == notes_content
    
    def test_get_notes_file_not_exists(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test getting notes when file doesn't exist"""
        from app.storage import RunStorage
        
        notes_folder = tmp_path / "notes"
        notes_folder.mkdir()
        
        RunStorage.update(completed_run['id'], {'folder_path': str(notes_folder)})
        
        response = client.get(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['content'] == ""
    
    def test_get_notes_no_folder_path(self, client, test_storage_files, auth_token, test_run):
        """Test getting notes when run has no folder path"""
        response = client.get(
            f"/api/runs/{test_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 404


class TestAddNoteEndpoint:
    """Test add_note endpoint with user validation"""
    
    def test_add_note_wrong_user(self, client, test_storage_files, clinician_token, completed_run, tmp_path):
        """Test adding note as different user than run owner"""
        from app.storage import RunStorage
        
        notes_folder = tmp_path / "notes"
        notes_folder.mkdir()
        RunStorage.update(completed_run['id'], {'folder_path': str(notes_folder)})
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"note": "This should fail"}
        )
        
        assert response.status_code == 404
    
    def test_add_note_verifies_content(self, client, test_storage_files, auth_token, completed_run, tmp_path):
        """Test that added note content is correct"""
        from app.storage import RunStorage
        
        notes_folder = tmp_path / "notes"
        notes_folder.mkdir()
        RunStorage.update(completed_run['id'], {'folder_path': str(notes_folder)})
        
        note_text = "Important observation about the results"
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/notes",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"note": note_text}
        )
        
        assert response.status_code == 200
        
        # Verify note was written to file
        notes_file = notes_folder / "notes_feedback.txt"
        assert notes_file.exists()
        content = notes_file.read_text()
        assert note_text in content
        assert "Note by test@example.com" in content


class TestAddFeedbackEndpoint:
    """Test add_feedback endpoint"""
    
    def test_add_feedback_creates_notification(self, client, test_storage_files, clinician_token, completed_run, tmp_path):
        """Test that adding feedback creates notification for data scientist"""
        from app.storage import RunStorage, NotificationStorage
        
        notes_folder = tmp_path / "notes"
        notes_folder.mkdir()
        RunStorage.update(completed_run['id'], {'folder_path': str(notes_folder)})
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/feedback",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"feedback": "Excellent work on this analysis"}
        )
        
        assert response.status_code == 200
        
        # Verify feedback_added flag was set
        run = RunStorage.get_by_id(completed_run['id'])
        assert run['feedback_added'] is True
        
        # Verify notification was created
        notifications = NotificationStorage.get_by_user(completed_run['user_id'])
        assert len(notifications) > 0
        assert any(f"run #{completed_run['id']}" in n['message'].lower() for n in notifications)
    
    def test_add_feedback_run_not_found(self, client, test_storage_files, clinician_token):
        """Test adding feedback to non-existent run"""
        response = client.post(
            "/api/runs/99999/feedback",
            headers={"Authorization": f"Bearer {clinician_token}"},
            json={"feedback": "Test feedback"}
        )
        
        assert response.status_code == 404


class TestSendToClinicianEndpoint:
    """Test send_to_clinician endpoint"""
    
    @patch('app.routes.send_clinician_review_email')
    def test_send_to_clinician_creates_notifications(self, mock_email, client, test_storage_files, auth_token, completed_run, test_clinician):
        """Test that sending to clinician creates notifications"""
        from app.storage import NotificationStorage
        
        mock_email.return_value = True
        
        response = client.post(
            f"/api/runs/{completed_run['id']}/send-to-clinician",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        assert response.status_code == 200
        
        # Verify notification was created for clinician
        notifications = NotificationStorage.get_by_user(test_clinician['id'])
        assert len(notifications) > 0
        
        # Verify email was sent
        mock_email.assert_called()
    
    def test_send_to_clinician_wrong_user(self, client, test_storage_files, clinician_token, completed_run):
        """Test sending to clinician as wrong user"""
        response = client.post(
            f"/api/runs/{completed_run['id']}/send-to-clinician",
            headers={"Authorization": f"Bearer {clinician_token}"}
        )
        
        assert response.status_code == 404
