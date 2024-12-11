#!/bin/bash

echo "=== Running Code Quality Checks ==="
echo

echo "=== Flake8 Check ==="
flake8 . --count --statistics --show-source --max-line-length=100
echo

echo "=== Pylint Check ==="
pylint --rcfile=setup.cfg $(git ls-files '*.py')
echo

echo "=== Type Checking with mypy ==="
mypy --ignore-missing-imports .
echo

echo "=== All checks completed ==="