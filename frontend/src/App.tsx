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
        main: darkMode ? "#33d1cc" : "#0f766e",
      },
      secondary: {
        main: darkMode ? "#5bc0ff" : "#0369a1",
      },
      success: {
        main: darkMode ? "#43c16f" : "#2e7d32",
      },
      warning: {
        main: darkMode ? "#ffb347" : "#ed6c02",
      },
      error: {
        main: darkMode ? "#ff5b6e" : "#d32f2f",
      },
      background: {
        default: darkMode ? "#040a17" : "#e6f5ff",
        paper: darkMode ? "rgba(9, 24, 47, 0.72)" : "rgba(255, 255, 255, 0.85)",
      },
      text: {
        primary: darkMode ? "#ebf4ff" : "#10213f",
        secondary: darkMode ? "#9bb4d8" : "#3f5f87",
      },
    },
    shape: {
      borderRadius: 16,
    },
    typography: {
      fontFamily: '"Space Grotesk", "Segoe UI", sans-serif',
      h5: {
        fontWeight: 700,
        letterSpacing: "0.01em",
      },
      h6: {
        fontWeight: 700,
      },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: darkMode ? "#040a17" : "#e6f5ff",
            backgroundImage: darkMode
              ? "radial-gradient(circle at 12% 8%, rgba(41, 138, 255, 0.24) 0%, rgba(41, 138, 255, 0) 38%), radial-gradient(circle at 82% 22%, rgba(51, 209, 204, 0.22) 0%, rgba(51, 209, 204, 0) 42%), linear-gradient(140deg, #040a17 0%, #071124 55%, #040a17 100%)"
              : "radial-gradient(circle at 18% 10%, rgba(18, 148, 235, 0.2) 0%, rgba(18, 148, 235, 0) 40%), radial-gradient(circle at 84% 18%, rgba(20, 184, 166, 0.2) 0%, rgba(20, 184, 166, 0) 36%), linear-gradient(155deg, #ecf7ff 0%, #f5fbff 45%, #ebf6ff 100%)",
            backgroundAttachment: "fixed",
          },
          "#root": {
            minHeight: "100vh",
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            background: darkMode
              ? "linear-gradient(98deg, rgba(8, 21, 44, 0.9) 0%, rgba(6, 34, 70, 0.9) 55%, rgba(8, 30, 56, 0.9) 100%)"
              : "linear-gradient(98deg, rgba(255, 255, 255, 0.9) 0%, rgba(236, 247, 255, 0.95) 100%)",
            borderBottom: darkMode
              ? "1px solid rgba(132, 176, 255, 0.18)"
              : "1px solid rgba(35, 94, 180, 0.18)",
            boxShadow: darkMode
              ? "0 12px 30px rgba(1, 6, 18, 0.5)"
              : "0 10px 24px rgba(36, 96, 168, 0.18)",
            backdropFilter: "blur(10px)",
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: darkMode
              ? "linear-gradient(150deg, rgba(14, 34, 67, 0.88) 0%, rgba(8, 23, 45, 0.82) 100%)"
              : "linear-gradient(145deg, rgba(255, 255, 255, 0.88) 0%, rgba(246, 251, 255, 0.95) 100%)",
            border: darkMode
              ? "1px solid rgba(104, 151, 241, 0.2)"
              : "1px solid rgba(95, 141, 224, 0.2)",
            boxShadow: darkMode
              ? "0 14px 34px rgba(2, 8, 24, 0.4), inset 0 1px 0 rgba(166, 205, 255, 0.08)"
              : "0 14px 30px rgba(60, 118, 193, 0.16), inset 0 1px 0 rgba(255, 255, 255, 0.7)",
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontWeight: 600,
            backdropFilter: "blur(4px)",
          },
        },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            backgroundColor: darkMode
              ? "rgba(7, 20, 40, 0.72)"
              : "rgba(252, 254, 255, 0.9)",
            "& .MuiOutlinedInput-notchedOutline": {
              borderColor: darkMode
                ? "rgba(121, 161, 233, 0.28)"
                : "rgba(80, 122, 205, 0.25)",
            },
            "&:hover .MuiOutlinedInput-notchedOutline": {
              borderColor: darkMode
                ? "rgba(130, 192, 255, 0.45)"
                : "rgba(39, 124, 243, 0.4)",
            },
            "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
              borderColor: darkMode ? "#33d1cc" : "#0ea5a5",
              boxShadow: darkMode
                ? "0 0 0 3px rgba(51, 209, 204, 0.18)"
                : "0 0 0 3px rgba(14, 165, 165, 0.15)",
            },
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          containedPrimary: {
            background: darkMode
              ? "linear-gradient(92deg, #2dbeb7 0%, #42e0cc 100%)"
              : "linear-gradient(92deg, #0f766e 0%, #14b8a6 100%)",
            color: darkMode ? "#042126" : "#f2fffd",
            boxShadow: darkMode
              ? "0 10px 26px rgba(45, 190, 183, 0.35)"
              : "0 10px 22px rgba(15, 118, 110, 0.25)",
          },
        },
      },
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
      <AppBar position="static" elevation={0}>
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
                sx={{
                  mr: 2,
                  boxShadow: health.model_loaded
                    ? "0 0 16px rgba(67, 193, 111, 0.4)"
                    : "0 0 14px rgba(255, 91, 110, 0.32)",
                }}
              />
              <Chip
                label={`Uptime: ${Math.floor(health.uptime_seconds / 60)}m`}
                variant="outlined"
                size="small"
                sx={{
                  mr: 2,
                  borderColor: "rgba(131, 171, 236, 0.35)",
                }}
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
            <Paper elevation={0} sx={{ p: 3 }}>
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
              <Paper elevation={0} sx={{ p: 3 }}>
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
                <Paper elevation={0} sx={{ p: 3 }}>
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
            <Paper elevation={0} sx={{ p: 3 }}>
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
            <Paper elevation={0} sx={{ p: 3 }}>
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
            Telco Churn Prediction System
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
