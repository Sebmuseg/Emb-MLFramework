#!/bin/bash

# Define the folder where Docker Compose resides
DEPLOYMENT_FOLDER="."

# Optional flags
REBUILD=false
CLEANUP=false

# Function to show usage instructions
show_usage() {
    echo "Usage: $0 [-r] [-c]"
    echo "  -r: Rebuild Docker images (useful if Dockerfile or dependencies have changed)."
    echo "  -c: Clean up unused Docker resources after running the application."
}

# Parse command-line options
while getopts "rc" flag; do
    case "${flag}" in
        r) REBUILD=true ;;
        c) CLEANUP=true ;;
        *) show_usage; exit 1 ;;
    esac
done

# Navigate to the deployment folder
cd "$DEPLOYMENT_FOLDER" || { echo "Deployment folder not found!"; exit 1; }

# Bring up Docker Compose with or without rebuilding
if [ "$REBUILD" = true ]; then
    echo "Rebuilding Docker images..."
    docker compose -f docker-compose.yml -p featherml_deployment up --build
else
    echo "Starting Docker Compose environment..."
    docker compose -f docker-compose.yml -p featherml_deployment up
fi

# Optional cleanup of unused Docker resources
if [ "$CLEANUP" = true ]; then
    echo "Cleaning up unused Docker resources..."
    docker system prune -a --volumes -f
fi