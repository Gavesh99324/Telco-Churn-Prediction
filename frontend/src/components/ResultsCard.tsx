import {
  Box,
  Typography,
  CircularProgress,
  Chip,
  Grid,
  LinearProgress,
  Alert,
  Card,
  CardContent,
} from "@mui/material";
import {
  CheckCircle as CheckIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from "@mui/icons-material";
import { PredictionResponse } from "../services/api";

interface ResultsCardProps {
  prediction: PredictionResponse | null;
  loading: boolean;
}

export default function ResultsCard({ prediction, loading }: ResultsCardProps) {
  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          py: 8,
        }}
      >
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (!prediction) {
    return (
      <Box sx={{ textAlign: "center", py: 8 }}>
        <Typography variant="body1" color="text.secondary">
          Enter customer information and click "Predict Churn" to see results
        </Typography>
      </Box>
    );
  }

  const isChurn = prediction.prediction === 1;
  const probability = prediction.probability * 100;

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case "high":
        return "error";
      case "medium":
        return "warning";
      default:
        return "success";
    }
  };

  const getConfidenceIcon = (confidence: string) => {
    switch (confidence.toLowerCase()) {
      case "high":
        return <CheckIcon />;
      case "medium":
        return <WarningIcon />;
      default:
        return <ErrorIcon />;
    }
  };

  return (
    <Box>
      {/* Main Prediction Result */}
      <Card
        sx={{
          mb: 3,
          bgcolor: isChurn ? "error.dark" : "success.dark",
          color: "white",
          textAlign: "center",
        }}
      >
        <CardContent>
          <Typography
            variant="h3"
            component="div"
            sx={{ fontWeight: "bold", mb: 1 }}
          >
            {isChurn ? "CHURN" : "NO CHURN"}
          </Typography>
          <Typography variant="h5">
            {probability.toFixed(1)}% Probability
          </Typography>
        </CardContent>
      </Card>

      {/* Probability Bar */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Churn Probability
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            {probability.toFixed(1)}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={probability}
          sx={{
            height: 10,
            borderRadius: 5,
            bgcolor: "grey.300",
            "& .MuiLinearProgress-bar": {
              bgcolor: isChurn ? "error.main" : "success.main",
            },
          }}
        />
      </Box>

      {/* Risk & Confidence Metrics */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Risk Level
            </Typography>
            <Chip
              label={prediction.risk_level}
              color={getRiskColor(prediction.risk_level)}
              sx={{ fontWeight: "bold", fontSize: "0.9rem", width: "100%" }}
            />
          </Box>
        </Grid>
        <Grid item xs={6}>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Confidence
            </Typography>
            <Chip
              icon={getConfidenceIcon(prediction.confidence)}
              label={prediction.confidence}
              color={getRiskColor(prediction.confidence)}
              sx={{ fontWeight: "bold", fontSize: "0.9rem", width: "100%" }}
            />
          </Box>
        </Grid>
      </Grid>

      {/* Recommendation Alert */}
      {isChurn && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="body2" fontWeight="bold">
            High churn risk detected!
          </Typography>
          <Typography variant="caption">
            Consider retention strategies: special offers, loyalty programs, or
            customer outreach.
          </Typography>
        </Alert>
      )}

      {!isChurn && (
        <Alert severity="success" sx={{ mb: 2 }}>
          <Typography variant="body2" fontWeight="bold">
            Customer retention likely!
          </Typography>
          <Typography variant="caption">
            Customer shows strong loyalty indicators. Continue excellent
            service.
          </Typography>
        </Alert>
      )}

      {/* Metadata */}
      <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: "divider" }}>
        <Grid container spacing={1}>
          <Grid item xs={12}>
            <Typography variant="caption" color="text.secondary">
              Prediction Time: {new Date(prediction.timestamp).toLocaleString()}
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Typography variant="caption" color="text.secondary">
              Model Version: {prediction.model_version}
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
}
