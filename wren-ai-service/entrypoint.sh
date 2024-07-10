#!/bin/bash
set -e

# Start wren-ai-service in the background
uvicorn src.__main__:app --host 0.0.0.0 --port $WREN_AI_SERVICE_PORT --loop uvloop --http httptools &

# Wait for the server to be responsive
echo "Waiting for wren-ai-service to start..."
while ! nc -z localhost $WREN_AI_SERVICE_PORT; do   
    sleep 1  # wait for 1 second before check again
done

echo "wren-ai-service has started."

python -m src.force_deploy

# Bring wren-ai-service to the foreground
wait