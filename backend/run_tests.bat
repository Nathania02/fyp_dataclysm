@echo off
echo ================================
echo Running Backend Unit Tests
echo ================================
echo.

REM Check if pytest is installed
python -m pytest --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pytest is not installed. Installing test dependencies...
    pip install -r requirements-test.txt
)

echo Running tests with coverage...
echo.

REM Run tests with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

echo.
echo ================================
echo Test run complete!
echo ================================
echo.
echo Coverage report generated in htmlcov/index.html
echo Open htmlcov/index.html in a browser to view detailed coverage report.
echo.

pause
