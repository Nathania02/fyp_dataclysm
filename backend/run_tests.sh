#!/bin/bash
# Run all tests with coverage

echo "================================"
echo "Running Backend Unit Tests"
echo "================================"
echo

# Check if pytest is installed
if ! python -m pytest --version &> /dev/null; then
    echo "pytest is not installed. Installing test dependencies..."
    pip install -r requirements-test.txt
fi

echo "Running tests with coverage..."
echo

# Run tests with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

echo
echo "================================"
echo "Test run complete!"
echo "================================"
echo
echo "Coverage report generated in htmlcov/index.html"
echo "Open htmlcov/index.html in a browser to view detailed coverage report."
echo
