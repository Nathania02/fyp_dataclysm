"""
Unit tests for storage functionality
"""
import pytest
from app.storage import UserStorage, RunStorage, NotificationStorage
from app.auth import get_password_hash

class TestUserStorage:
    """Test UserStorage operations"""
    
    def test_create_user(self, test_storage_files):
        """Test creating a new user"""
        user_data = {
            'email': 'newuser@example.com',
            'hashed_password': get_password_hash('password123'),
            'role': 'data_scientist'
        }
        
        user = UserStorage.create(user_data)
        
        assert user['id'] == 1
        assert user['email'] == 'newuser@example.com'
        assert user['role'] == 'data_scientist'
        assert 'created_at' in user
    
    def test_get_user_by_email(self, test_storage_files, test_user):
        """Test retrieving user by email"""
        user = UserStorage.get_by_email('test@example.com')
        
        assert user is not None
        assert user['email'] == 'test@example.com'
        assert user['id'] == test_user['id']
    
    def test_get_user_by_id(self, test_storage_files, test_user):
        """Test retrieving user by ID"""
        user = UserStorage.get_by_id(test_user['id'])
        
        assert user is not None
        assert user['id'] == test_user['id']
        assert user['email'] == test_user['email']
    
    def test_get_all_users(self, test_storage_files, test_user, test_clinician):
        """Test retrieving all users"""
        users = UserStorage.get_all()
        
        assert len(users) == 2
        assert any(u['email'] == 'test@example.com' for u in users)
        assert any(u['email'] == 'clinician@example.com' for u in users)
    
    def test_get_nonexistent_user_by_email(self, test_storage_files):
        """Test retrieving non-existent user by email"""
        user = UserStorage.get_by_email('nonexistent@example.com')
        
        assert user is None
    
    def test_get_nonexistent_user_by_id(self, test_storage_files):
        """Test retrieving non-existent user by ID"""
        user = UserStorage.get_by_id(999)
        
        assert user is None
    
    def test_sequential_user_ids(self, test_storage_files):
        """Test that user IDs are sequential"""
        user1_data = {
            'email': 'user1@example.com',
            'hashed_password': 'hash1',
            'role': 'data_scientist'
        }
        user2_data = {
            'email': 'user2@example.com',
            'hashed_password': 'hash2',
            'role': 'clinician'
        }
        
        user1 = UserStorage.create(user1_data)
        user2 = UserStorage.create(user2_data)
        
        assert user2['id'] == user1['id'] + 1


class TestRunStorage:
    """Test RunStorage operations"""
    
    def test_create_run(self, test_storage_files, test_user):
        """Test creating a new run"""
        run_data = {
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        }
        
        run = RunStorage.create(run_data)
        
        assert run['id'] == 1
        assert run['user_id'] == test_user['id']
        assert run['model_type'] == 'kmeans'
        assert run['status'] == 'pending'
        assert run['optimal_clusters'] is None
        assert run['sent_to_clinician'] is False
        assert run['feedback_added'] is False
    
    def test_get_run_by_id(self, test_storage_files, test_run):
        """Test retrieving run by ID"""
        run = RunStorage.get_by_id(test_run['id'])
        
        assert run is not None
        assert run['id'] == test_run['id']
        assert run['model_type'] == test_run['model_type']
    
    def test_get_runs_by_user(self, test_storage_files, test_user, test_clinician):
        """Test retrieving runs by user"""
        # Create runs for different users
        run1_data = {
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test1.duckdb',
            'dataset_name': 'test_table1',
            'parameters_filename': 'params1.yaml'
        }
        run2_data = {
            'user_id': test_clinician['id'],
            'user_email': test_clinician['email'],
            'model_type': 'lca',
            'dataset_filename': 'test2.duckdb',
            'dataset_name': 'test_table2',
            'parameters_filename': 'params2.yaml'
        }
        
        RunStorage.create(run1_data)
        RunStorage.create(run2_data)
        
        user_runs = RunStorage.get_by_user(test_user['id'])
        
        assert len(user_runs) == 1
        assert user_runs[0]['user_id'] == test_user['id']
    
    def test_get_all_runs(self, test_storage_files, test_user):
        """Test retrieving all runs"""
        run1_data = {
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'kmeans',
            'dataset_filename': 'test1.duckdb',
            'dataset_name': 'test_table1',
            'parameters_filename': 'params1.yaml'
        }
        run2_data = {
            'user_id': test_user['id'],
            'user_email': test_user['email'],
            'model_type': 'lca',
            'dataset_filename': 'test2.duckdb',
            'dataset_name': 'test_table2',
            'parameters_filename': 'params2.yaml'
        }
        
        RunStorage.create(run1_data)
        RunStorage.create(run2_data)
        
        runs = RunStorage.get_all()
        
        assert len(runs) == 2
    
    def test_update_run(self, test_storage_files, test_run):
        """Test updating a run"""
        updates = {
            'status': 'completed',
            'optimal_clusters': 3,
            'completed_at': '2025-01-01T00:00:00'
        }
        
        updated_run = RunStorage.update(test_run['id'], updates)
        
        assert updated_run['status'] == 'completed'
        assert updated_run['optimal_clusters'] == 3
        assert updated_run['completed_at'] == '2025-01-01T00:00:00'
    
    def test_update_nonexistent_run(self, test_storage_files):
        """Test updating non-existent run"""
        result = RunStorage.update(999, {'status': 'completed'})
        
        assert result is None


class TestNotificationStorage:
    """Test NotificationStorage operations"""
    
    def test_create_notification(self, test_storage_files, test_user, test_run):
        """Test creating a new notification"""
        notification_data = {
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Test notification'
        }
        
        notification = NotificationStorage.create(notification_data)
        
        assert notification['id'] == 1
        assert notification['user_id'] == test_user['id']
        assert notification['run_id'] == test_run['id']
        assert notification['message'] == 'Test notification'
        assert notification['is_read'] is False
        assert 'created_at' in notification
    
    def test_get_notifications_by_user(self, test_storage_files, test_user, test_clinician, test_run):
        """Test retrieving notifications by user"""
        # Create notifications for different users
        NotificationStorage.create({
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Notification 1'
        })
        NotificationStorage.create({
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Notification 2'
        })
        NotificationStorage.create({
            'user_id': test_clinician['id'],
            'run_id': test_run['id'],
            'message': 'Notification 3'
        })
        
        user_notifications = NotificationStorage.get_by_user(test_user['id'])
        
        assert len(user_notifications) == 2
        assert all(n['user_id'] == test_user['id'] for n in user_notifications)
    
    def test_get_notification_by_id(self, test_storage_files, test_notification):
        """Test retrieving notification by ID"""
        notification = NotificationStorage.get_by_id(test_notification['id'])
        
        assert notification is not None
        assert notification['id'] == test_notification['id']
        assert notification['message'] == test_notification['message']
    
    def test_update_notification(self, test_storage_files, test_notification):
        """Test updating a notification"""
        updated = NotificationStorage.update(test_notification['id'], {'is_read': True})
        
        assert updated['is_read'] is True
    
    def test_get_all_notifications(self, test_storage_files, test_user, test_run):
        """Test retrieving all notifications"""
        NotificationStorage.create({
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Notification 1'
        })
        NotificationStorage.create({
            'user_id': test_user['id'],
            'run_id': test_run['id'],
            'message': 'Notification 2'
        })
        
        notifications = NotificationStorage.get_all()
        
        assert len(notifications) == 2
