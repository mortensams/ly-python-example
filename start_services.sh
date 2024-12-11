#!/bin/bash

# Start the FastAPI server in the background
python api.py &

# Wait for API to be ready
echo "Waiting for API to start..."
until $(curl --output /dev/null --silent --head --fail http://localhost:8000/health); do
    printf '.'
    sleep 1
done
echo "API is ready!"

# Start the temperature viewer
echo "Starting temperature viewer..."
python temperature_viewer.py