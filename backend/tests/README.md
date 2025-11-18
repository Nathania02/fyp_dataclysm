# Backend Testing Guide

## Overview
This directory contains comprehensive unit tests for the backend application, covering all major features including authentication, storage, API routes, and schema validation.

## Test Structure

```
tests/
├── __init__.py              # Package marker
├── conftest.py              # Pytest fixtures and configuration
├── test_auth.py             # Authentication tests
├── test_storage.py          # Storage layer tests
├── test_routes.py           # API route tests
└── test_schemas.py          # Schema validation tests
```

## Test Coverage

### 1. Authentication Tests (`test_auth.py`)
- Password hashing and verification
- JWT token creation and validation
- User authentication and authorization
- Invalid token handling

### 2. Storage Tests (`test_storage.py`)
- User CRUD operations
- Run CRUD operations
- Notification CRUD operations
- Data persistence and retrieval

### 3. Route Tests (`test_routes.py`)
- User signup and login
- Run creation and management
- Run status updates
- Notes and feedback
- Notifications
- Authorization checks
- Clinician workflows

### 4. Schema Tests (`test_schemas.py`)
- Pydantic model validation
- Enum validations
- Required vs optional fields
- Serialization/deserialization

## Running Tests

### Install Dependencies
```bash
pip install pytest pytest-cov pytest-asyncio httpx
```

### Run All Tests
```bash
pytest
```

### Run Specific Test File
```bash
pytest tests/test_auth.py
```

### Run Specific Test Class
```bash
pytest tests/test_auth.py::TestPasswordHashing
```

### Run Specific Test
```bash
pytest tests/test_auth.py::TestPasswordHashing::test_password_hashing
```

### Run with Coverage Report
```bash
pytest --cov=app --cov-report=html
```

### Run Verbose Mode
```bash
pytest -v
```

### Run and Stop on First Failure
```bash
pytest -x
```

### Run Tests in Parallel (faster)
```bash
pip install pytest-xdist
pytest -n auto
```

## Test Fixtures

### Storage Fixtures
- `test_storage_files`: Creates temporary JSON storage files for isolated testing
- `test_user`: Creates a data scientist test user
- `test_clinician`: Creates a clinician test user
- `test_run`: Creates a test model run
- `completed_run`: Creates a completed test run
- `test_notification`: Creates a test notification

### Authentication Fixtures
- `auth_token`: JWT token for authenticated test user
- `clinician_token`: JWT token for clinician test user
- `client`: FastAPI test client

## Writing New Tests

### Example Test Structure
```python
class TestNewFeature:
    """Test new feature"""
    
    def test_feature_success(self, client, auth_token):
        """Test successful feature execution"""
        response = client.post(
            "/api/new-feature",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"data": "test"}
        )
        
        assert response.status_code == 200
        assert response.json()['status'] == 'success'
    
    def test_feature_failure(self, client):
        """Test feature failure scenario"""
        response = client.post("/api/new-feature")
        
        assert response.status_code == 401
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on other tests
2. **Fixtures**: Use fixtures for common setup to avoid code duplication
3. **Clear Names**: Test names should clearly describe what is being tested
4. **Assert Messages**: Include descriptive assertion messages
5. **Mock External Services**: Use mocks for email services, external APIs, etc.
6. **Test Both Success and Failure**: Test both happy paths and error cases
7. **Clean Up**: Fixtures handle cleanup automatically, but ensure no side effects

## Continuous Integration

Tests should be run automatically on:
- Every commit
- Every pull request
- Before deployment

Example GitHub Actions workflow:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov pytest-asyncio
      - run: pytest --cov=app
```

## Troubleshooting

### Tests Fail Due to File Permissions
- Ensure test storage files are cleaned up properly
- Check the `test_storage_files` fixture

### Import Errors
- Ensure all dependencies are installed
- Check PYTHONPATH includes the backend directory

### Async Test Errors
- Ensure pytest-asyncio is installed
- Use `@pytest.mark.asyncio` for async tests

### Mock Not Working
- Verify the correct module path in `@patch` decorator
- Ensure mock is set up before the function is called

## Coverage Goals

Aim for:
- **Overall Coverage**: > 80%
- **Critical Paths**: 100% (auth, data storage)
- **Business Logic**: > 90%
- **Route Handlers**: > 85%

## Current Test Statistics

Run `pytest --cov=app` to see current coverage statistics.

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain or improve coverage
4. Document any new fixtures or test utilities
