import { useState, useEffect } from "react";
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Grid,
  Paper,
  Alert,
  Chip,
  IconButton,
} from "@mui/material";
import {
  Brightness4 as DarkModeIcon,
  Brightness7 as LightModeIcon,
  Psychology as BrainIcon,
} from "@mui/icons-material";
import PredictionForm from "./components/PredictionForm";
import ResultsCard from "./components/ResultsCard";
import MetricsDashboard from "./components/MetricsDashboard";
import HistoryTable from "./components/HistoryTable";
import { PredictionResponse, HealthResponse, ModelInfo } from "./services/api";
import { apiService } from "./services/api";
import { PredictionHistory } from "./types";

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [history, setHistory] = useState<PredictionHistory[]>([]);

  const theme = createTheme({
    palette: {
      mode: darkMode ? "dark" : "light",
      primary: {
        main: "#1976d2",
      },
      secondary: {
        main: "#dc004e",
      },
      success: {
        main: "#4caf50",
      },
      warning: {
        main: "#ff9800",
      },
      error: {
        main: "#f44336",
      },
    },
    typography: {
      fontFamily: "Roboto, Arial, sans-serif",
    },
  });

  // Check health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const healthData = await apiService.health();
        setHealth(healthData);

        if (healthData.model_loaded) {
          const modelData = await apiService.getModelInfo();
          setModelInfo(modelData);
        }
      } catch (err) {
        console.error("Health check failed:", err);
        setError(
          "Unable to connect to API server. Please ensure the backend is running.",
        );
      }
    };

    checkHealth();

    // Poll health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePredict = async (customerData: any) => {
    setLoading(true);
    setError(null);

    try {
      const result = await apiService.predictChurn(customerData);
      setPrediction(result);

      // Add to history
      const historyItem: PredictionHistory = {
        id: Date.now().toString(),
        timestamp: result.timestamp,
        customerInfo: `Tenure: ${customerData.tenure} months, Contract: ${customerData.Contract}`,
        prediction: result.prediction,
        probability: result.probability,
        riskLevel: result.risk_level,
        confidence: result.confidence,
      };

      setHistory((prev) => [historyItem, ...prev].slice(0, 50)); // Keep last 50
    } catch (err: any) {
      console.error("Prediction error:", err);
      setError(
        err.response?.data?.detail || "Prediction failed. Please try again.",
      );
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* App Bar */}
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <BrainIcon sx={{ mr: 2, fontSize: 32 }} />
          <Typography
            variant="h5"
            component="div"
            sx={{ flexGrow: 1, fontWeight: 600 }}
          >
            Telco Churn Prediction Dashboard
          </Typography>

          {/* Status Indicators */}
          {health && (
            <>
              <Chip
                label={health.model_loaded ? "Model Ready" : "Model Not Loaded"}
                color={health.model_loaded ? "success" : "error"}
                size="small"
                sx={{ mr: 2 }}
              />
              <Chip
                label={`Uptime: ${Math.floor(health.uptime_seconds / 60)}m`}
                variant="outlined"
                size="small"
                sx={{ mr: 2 }}
              />
            </>
          )}

          <IconButton onClick={toggleDarkMode} color="inherit">
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        {/* Error Alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* API Status Alert */}
        {!health && (
          <Alert severity="warning" sx={{ mb: 3 }}>
            Connecting to API server...
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Left Column - Prediction Form */}
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ mb: 3, fontWeight: 600 }}
              >
                Customer Information
              </Typography>
              <PredictionForm
                onPredict={handlePredict}
                loading={loading}
                disabled={!health?.model_loaded}
              />
            </Paper>
          </Grid>

          {/* Right Column - Results */}
          <Grid item xs={12} md={6}>
            <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {/* Prediction Results */}
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography
                  variant="h6"
                  gutterBottom
                  sx={{ mb: 3, fontWeight: 600 }}
                >
                  Prediction Results
                </Typography>
                <ResultsCard prediction={prediction} loading={loading} />
              </Paper>

              {/* Model Info */}
              {modelInfo && (
                <Paper elevation={3} sx={{ p: 3 }}>
                  <Typography
                    variant="h6"
                    gutterBottom
                    sx={{ mb: 2, fontWeight: 600 }}
                  >
                    Model Information
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Model Name
                      </Typography>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {modelInfo.model_name}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Version
                      </Typography>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {modelInfo.version}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Features
                      </Typography>
                      <Typography variant="body1" sx={{ fontWeight: 500 }}>
                        {modelInfo.feature_count}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Status
                      </Typography>
                      <Chip
                        label="Active"
                        color="success"
                        size="small"
                        sx={{ mt: 0.5 }}
                      />
                    </Grid>
                  </Grid>
                </Paper>
              )}
            </Box>
          </Grid>

          {/* Metrics Dashboard */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ mb: 3, fontWeight: 600 }}
              >
                Performance Metrics
              </Typography>
              <MetricsDashboard history={history} />
            </Paper>
          </Grid>

          {/* Prediction History */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography
                variant="h6"
                gutterBottom
                sx={{ mb: 3, fontWeight: 600 }}
              >
                Prediction History
              </Typography>
              <HistoryTable history={history} />
            </Paper>
          </Grid>
        </Grid>

        {/* Footer */}
        <Box sx={{ mt: 6, mb: 2, textAlign: "center" }}>
          <Typography variant="body2" color="text.secondary">
            Telco Churn Prediction System v1.0.0 | Powered by FastAPI + React +
            MUI
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
