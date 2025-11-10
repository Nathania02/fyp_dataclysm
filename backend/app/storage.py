import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.config import settings

# Initialize storage directory and files
os.makedirs(settings.RESULTS_DIR, exist_ok=True)

def _ensure_file(filepath: str, default_data: Any):
    """Ensure a JSON file exists with default data"""
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_data, f, indent=2)

def _read_json(filepath: str) -> Any:
    """Read JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def _write_json(filepath: str, data: Any):
    """Write JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

# Initialize files
_ensure_file(settings.USERS_FILE, [])
_ensure_file(settings.RUNS_FILE, [])
_ensure_file(settings.NOTIFICATIONS_FILE, [])

class UserStorage:
    @staticmethod
    def get_all() -> List[Dict]:
        return _read_json(settings.USERS_FILE)
    
    @staticmethod
    def get_by_email(email: str) -> Optional[Dict]:
        users = _read_json(settings.USERS_FILE)
        return next((u for u in users if u['email'] == email), None)
    
    @staticmethod
    def get_by_id(user_id: int) -> Optional[Dict]:
        users = _read_json(settings.USERS_FILE)
        return next((u for u in users if u['id'] == user_id), None)
    
    @staticmethod
    def create(user_data: Dict) -> Dict:
        users = _read_json(settings.USERS_FILE)
        new_id = max([u['id'] for u in users], default=0) + 1
        user = {
            'id': new_id,
            'email': user_data['email'],
            'hashed_password': user_data['hashed_password'],
            'role': user_data['role'],
            'created_at': datetime.utcnow().isoformat()
        }
        users.append(user)
        _write_json(settings.USERS_FILE, users)
        return user

class RunStorage:
    @staticmethod
    def get_all() -> List[Dict]:
        return _read_json(settings.RUNS_FILE)
    
    @staticmethod
    def get_by_id(run_id: int) -> Optional[Dict]:
        runs = _read_json(settings.RUNS_FILE)
        return next((r for r in runs if r['id'] == run_id), None)
    
    @staticmethod
    def get_by_user(user_id: int) -> List[Dict]:
        runs = _read_json(settings.RUNS_FILE)
        print(runs)
        return [r for r in runs if r['user_id'] == user_id]
    
    @staticmethod
    def create(run_data: Dict) -> Dict:
        runs = _read_json(settings.RUNS_FILE)
        new_id = max([r['id'] for r in runs], default=0) + 1
        run = {
            'id': new_id,
            'user_id': run_data['user_id'],
            'model_type': run_data['model_type'],
            'status': 'pending',
            'folder_path': None,
            'dataset_filename': run_data.get('dataset_filename'),
            'dataset_name': run_data.get('dataset_name'),
            'parameters_filename': run_data.get('parameters_filename'),
            'optimal_clusters': None,
            'created_at': datetime.utcnow().isoformat(),
            'completed_at': None,
            'celery_task_id': None,
            'sent_to_clinician': False,
            'feedback_added': False
        }
        runs.append(run)
        _write_json(settings.RUNS_FILE, runs)
        return run
    
    @staticmethod
    def update(run_id: int, updates: Dict) -> Optional[Dict]:
        runs = _read_json(settings.RUNS_FILE)
        for i, run in enumerate(runs):
            if run['id'] == run_id:
                runs[i].update(updates)
                _write_json(settings.RUNS_FILE, runs)
                return runs[i]
        return None

class NotificationStorage:
    @staticmethod
    def get_all() -> List[Dict]:
        return _read_json(settings.NOTIFICATIONS_FILE)
    
    @staticmethod
    def get_by_user(user_id: int) -> List[Dict]:
        notifications = _read_json(settings.NOTIFICATIONS_FILE)
        return [n for n in notifications if n['user_id'] == user_id]
    
    @staticmethod
    def get_by_id(notification_id: int) -> Optional[Dict]:
        notifications = _read_json(settings.NOTIFICATIONS_FILE)
        return next((n for n in notifications if n['id'] == notification_id), None)
    
    @staticmethod
    def create(notification_data: Dict) -> Dict:
        notifications = _read_json(settings.NOTIFICATIONS_FILE)
        new_id = max([n['id'] for n in notifications], default=0) + 1
        notification = {
            'id': new_id,
            'user_id': notification_data['user_id'],
            'run_id': notification_data['run_id'],
            'message': notification_data['message'],
            'is_read': False,
            'created_at': datetime.utcnow().isoformat()
        }
        notifications.append(notification)
        _write_json(settings.NOTIFICATIONS_FILE, notifications)
        return notification
    
    @staticmethod
    def update(notification_id: int, updates: Dict) -> Optional[Dict]:
        notifications = _read_json(settings.NOTIFICATIONS_FILE)
        for i, notification in enumerate(notifications):
            if notification['id'] == notification_id:
                notifications[i].update(updates)
                _write_json(settings.NOTIFICATIONS_FILE, notifications)
                return notifications[i]
        return None