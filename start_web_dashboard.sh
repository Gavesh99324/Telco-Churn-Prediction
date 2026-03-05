#!/bin/bash
# Quick Start Script for Web Dashboard
# Telco Churn Prediction - React + MUI + FastAPI

echo "========================================"
echo "Telco Churn Prediction Web Dashboard"
echo "React + Material-UI + FastAPI"
echo "========================================"
echo ""

# Check if Docker is running
echo "Checking Docker..."
if docker ps >/dev/null 2>&1; then
    echo "✓ Docker is running"
else
    echo "✗ Docker is not running. Please start Docker."
    exit 1
fi

# Check if artifacts exist
echo ""
echo "Checking model artifacts..."
if [ -f "artifacts/models/best_model.pkl" ]; then
    echo "✓ Model found: artifacts/models/best_model.pkl"
else
    echo "⚠ Model not found. You may need to train the model first."
    echo "  Run: python pipelines/training_pipeline.py"
fi

# Build and start services
echo ""
echo "Building and starting web services..."
echo "This may take 2-3 minutes on first run..."
echo ""

docker-compose -f docker-compose.web.yml up -d --build

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Web Dashboard Started Successfully!"
    echo "========================================"
    echo ""
    echo "🌐 Access Points:"
    echo "  Frontend Dashboard:  http://localhost:3001"
    echo "  FastAPI Backend:     http://localhost:8000"
    echo "  API Documentation:   http://localhost:8000/docs"
    echo "  API Health Check:    http://localhost:8000/health"
    echo ""
    echo "📊 Features:"
    echo "  • Interactive prediction form with Material-UI"
    echo "  • Real-time churn probability calculation"
    echo "  • Performance metrics dashboard with charts"
    echo "  • Prediction history table"
    echo "  • Dark/Light theme toggle"
    echo "  • REST API for integration"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  View logs:     docker-compose -f docker-compose.web.yml logs -f"
    echo "  Stop services: docker-compose -f docker-compose.web.yml down"
    echo "  Restart:       docker-compose -f docker-compose.web.yml restart"
    echo ""
else
    echo ""
    echo "✗ Failed to start services"
    echo "Check logs: docker-compose -f docker-compose.web.yml logs"
    exit 1
fi
