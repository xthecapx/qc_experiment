#!/bin/bash

# Start the container in detached mode
docker-compose -f docker-compose.yml up --build -d

# Wait a few seconds for the container to initialize
echo "Waiting for Jupyter to start..."
sleep 5

# Get the token from the logs
TOKEN=$(docker-compose logs | grep -o "token=[a-z0-9]*" | head -1)

# Construct the URL
echo "================================================================="
echo "Jupyter Lab is ready at: http://127.0.0.1:4321/lab?$TOKEN"
echo "================================================================="

# Open the browser (works on Linux, macOS, and Windows)
case "$(uname -s)" in
    Linux*)     xdg-open "http://127.0.0.1:4321/lab?$TOKEN";;
    Darwin*)    open -na "Google Chrome" --args --new-window "http://127.0.0.1:4321/lab?$TOKEN" || open -a "Safari" "http://127.0.0.1:4321/lab?$TOKEN";;
    *)          echo "Please open http://127.0.0.1:4321/lab?$TOKEN in your browser";;
esac
