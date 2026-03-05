# Quick Start Script for Web Dashboard
# Telco Churn Prediction - React + MUI + FastAPI

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Telco Churn Prediction Web Dashboard" -ForegroundColor Cyan
Write-Host "React + Material-UI + FastAPI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
} catch {
    Write-Host "Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if artifacts exist
Write-Host ""
Write-Host "Checking model artifacts..." -ForegroundColor Yellow
if (Test-Path "artifacts/models/best_model.pkl") {
    Write-Host "Model found: artifacts/models/best_model.pkl" -ForegroundColor Green
} else {
    Write-Host "Model not found. You may need to train the model first." -ForegroundColor Yellow
    Write-Host "  Run: python pipelines/training_pipeline.py" -ForegroundColor Gray
}

# Build and start services
Write-Host ""
Write-Host "Building and starting web services..." -ForegroundColor Yellow
Write-Host "This may take 2-3 minutes on first run..." -ForegroundColor Gray
Write-Host ""

docker-compose -f docker-compose.web.yml up -d --build

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Web Dashboard Started Successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access Points:" -ForegroundColor Cyan
    Write-Host "  Frontend Dashboard:  http://localhost:3001" -ForegroundColor White
    Write-Host "  FastAPI Backend:     http://localhost:8000" -ForegroundColor White
    Write-Host "  API Documentation:   http://localhost:8000/docs" -ForegroundColor White
    Write-Host "  API Health Check:    http://localhost:8000/health" -ForegroundColor White
    Write-Host ""
    Write-Host "Features:" -ForegroundColor Cyan
    Write-Host "  - Interactive prediction form with Material-UI" -ForegroundColor Gray
    Write-Host "  - Real-time churn probability calculation" -ForegroundColor Gray
    Write-Host "  - Performance metrics dashboard with charts" -ForegroundColor Gray
    Write-Host "  - Prediction history table" -ForegroundColor Gray
    Write-Host "  - Dark/Light theme toggle" -ForegroundColor Gray
    Write-Host "  - REST API for integration" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Useful Commands:" -ForegroundColor Cyan
    Write-Host "  View logs:     docker-compose -f docker-compose.web.yml logs -f" -ForegroundColor Gray
    Write-Host "  Stop services: docker-compose -f docker-compose.web.yml down" -ForegroundColor Gray
    Write-Host "  Restart:       docker-compose -f docker-compose.web.yml restart" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Opening dashboard in browser..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:3001"
} else {
    Write-Host ""
    Write-Host "Failed to start services" -ForegroundColor Red
    Write-Host "Check logs: docker-compose -f docker-compose.web.yml logs" -ForegroundColor Yellow
    exit 1
}
