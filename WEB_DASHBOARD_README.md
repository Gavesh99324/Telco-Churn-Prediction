# Web Dashboard - React + MUI + FastAPI

Complete web-based UI for Telco Churn Prediction with modern Material-UI design.

## 🎨 Features

### **Frontend (React + TypeScript + MUI)**

- 📊 **Interactive Prediction Form** - Customer data input with validation
- 🎯 **Real-time Results** - Instant churn probability and risk assessment
- 📈 **Performance Metrics** - Charts and statistics dashboard
- 📜 **Prediction History** - DataGrid table with searchable history
- 🌓 **Dark/Light Mode** - Theme toggle
- 📱 **Responsive Design** - Mobile-friendly Material Design

### **Backend (FastAPI)**

- ⚡ **REST API** - RESTful endpoints for predictions
- 📚 **Auto Documentation** - Swagger UI at `/docs`
- 🔍 **Health Checks** - `/health` and `/ready` endpoints
- 📊 **Prometheus Metrics** - `/metrics` endpoint
- 🔐 **CORS Enabled** - Cross-origin support
- 🧪 **Input Validation** - Pydantic models

---

## 🚀 Quick Start

### **Option 1: Docker (Recommended)**

```bash
# Build and start web services
docker-compose -f docker-compose.web.yml up -d --build

# Access services
# Frontend: http://localhost:3001
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### **Option 2: Local Development**

**Backend:**

```bash
# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
cd api
uvicorn main:app --reload --port 8000

# API will be available at http://localhost:8000
```

**Frontend:**

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev

# Frontend will be available at http://localhost:3001
```

---

## 📡 API Endpoints

### **Prediction Endpoints**

#### `POST /predict`

Predict churn for a single customer.

**Request:**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": 848.4
}
```

**Response:**

```json
{
  "prediction": 1,
  "probability": 0.8547,
  "confidence": "High",
  "risk_level": "High",
  "timestamp": "2026-03-05T10:30:00.000Z",
  "model_version": "1.0.0"
}
```

#### `POST /batch_predict`

Predict churn for multiple customers.

**Request:**

```json
{
  "customers": [
    {
      /* customer 1 data */
    },
    {
      /* customer 2 data */
    }
  ]
}
```

**Response:**

```json
{
  "predictions": [...],
  "total_customers": 2,
  "churn_count": 1,
  "churn_percentage": 50.0,
  "processing_time_ms": 45.23
}
```

### **Health & Monitoring**

- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /model/info` - Model metadata

### **Documentation**

- `GET /docs` - Swagger UI (Interactive API docs)
- `GET /redoc` - ReDoc (Alternative API docs)

---

## 🎨 Frontend Components

### **PredictionForm**

- Material-UI form with validation
- Grouped sections (Demographics, Services, Account)
- Dropdowns for categorical fields
- Number inputs for numerical fields
- Reset and Submit buttons

### **ResultsCard**

- Large prediction display (Churn/No Churn)
- Probability bar chart
- Risk level and confidence chips
- Actionable recommendations
- Metadata (timestamp, model version)

### **MetricsDashboard**

- Summary cards (total predictions, churn rate)
- Line chart (churn probability trend)
- Pie chart (churn distribution)
- Recharts integration

### **HistoryTable**

- MUI DataGrid with pagination
- Searchable and sortable columns
- Color-coded prediction chips
- Timestamp formatting

---

## 🛠️ Technology Stack

### **Backend**

- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Scikit-learn** - ML model serving
- **Prometheus Client** - Metrics

### **Frontend**

- **React 18** - UI library
- **TypeScript** - Type safety
- **Material-UI (MUI) v5** - Component library
- **Recharts** - Charts
- **Axios** - HTTP client
- **Vite** - Build tool

### **Infrastructure**

- **Docker** - Containerization
- **Nginx** - Frontend web server
- **Docker Compose** - Multi-container orchestration

---

## 📊 Architecture

```
┌─────────────────┐         ┌─────────────────┐
│  React Frontend │ ──API──▶│  FastAPI Backend│
│  (Port 3001)    │ ◀─JSON──│  (Port 8000)    │
└─────────────────┘         └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │  ML Model       │
                            │  (best_model)   │
                            └─────────────────┘
```

**Request Flow:**

1. User fills form in React UI
2. Frontend sends POST `/predict` to FastAPI
3. FastAPI preprocesses data (feature engineering)
4. Model makes prediction
5. FastAPI returns JSON response
6. Frontend displays results with charts

---

## 🔧 Configuration

### **Environment Variables**

**Backend (.env):**

```bash
API_PORT=8000
ENVIRONMENT=production
MODEL_PATH=/app/artifacts/models/best_model.pkl
```

**Frontend (.env):**

```bash
VITE_API_URL=http://localhost:8000
VITE_APP_TITLE=Telco Churn Prediction
VITE_APP_VERSION=1.0.0
```

### **API Configuration**

Edit `api/main.py` to customize:

- CORS origins
- Model path
- Feature engineering logic
- Response format

### **Frontend Theme**

Edit `frontend/src/App.tsx` to customize:

- Color palette
- Typography
- Component styles

---

## 🧪 Testing

### **Backend**

```bash
# Test API health
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_customer.json

# View API docs
open http://localhost:8000/docs
```

### **Frontend**

```bash
cd frontend

# Run linter
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

---

## 📦 Production Deployment

### **Build Optimizations**

- Frontend: Vite tree-shaking, minification, code splitting
- Backend: Multi-worker Uvicorn (2 workers default)
- Nginx: Gzip compression, static asset caching
- Docker: Multi-stage builds for smaller images

### **Performance**

- API response time: ~50-100ms per prediction
- Frontend bundle size: ~500KB gzipped
- Docker image sizes:
  - Frontend: ~25MB (nginx:alpine base)
  - Backend: ~300MB (Python dependencies)

### **Monitoring**

- Prometheus metrics at `/metrics`
- Health checks every 30s
- Error logging to stdout (JSON format)

---

## 📸 Screenshots

### Dashboard Overview

- **Left Panel**: Customer input form with all 19 features
- **Right Panel**: Prediction results with probability, risk level, confidence
- **Bottom**: Metrics charts and prediction history table

### Key Features

- ✅ Material Design components
- ✅ Responsive layout (mobile-friendly)
- ✅ Dark/Light theme toggle
- ✅ Real-time validation
- ✅ Interactive charts
- ✅ Searchable data grid

---

## 🐛 Troubleshooting

### **API Not Starting**

```bash
# Check if model exists
ls -la artifacts/models/best_model.pkl

# Check logs
docker logs telco-api

# Rebuild container
docker-compose -f docker-compose.web.yml up -d --build api
```

### **Frontend Build Fails**

```bash
# Clear node_modules
cd frontend
rm -rf node_modules package-lock.json
npm install

# Check Node version (requires 16+)
node --version
```

### **CORS Errors**

Update `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # Add specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🚀 Next Steps

1. **Start Web Services:**

   ```bash
   docker-compose -f docker-compose.web.yml up -d
   ```

2. **Access Dashboard:**
   - Open http://localhost:3001 in browser

3. **Make Predictions:**
   - Fill customer data form
   - Click "Predict Churn"
   - View results and metrics

4. **Explore API:**
   - Visit http://localhost:8000/docs
   - Try interactive API testing

5. **Monitor Performance:**
   - Check prediction history table
   - View metrics dashboard charts

---

## 📚 Additional Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Material-UI Docs**: https://mui.com
- **React Docs**: https://react.dev
- **Vite Docs**: https://vitejs.dev

---

**Built with ❤️ using React + Material-UI + FastAPI**
