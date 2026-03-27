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
          background: isChurn
            ? "linear-gradient(120deg, rgba(123, 18, 32, 0.95) 0%, rgba(199, 41, 65, 0.88) 100%)"
            : "linear-gradient(120deg, rgba(23, 88, 44, 0.94) 0%, rgba(60, 171, 102, 0.9) 100%)",
          border: isChurn
            ? "1px solid rgba(255, 159, 174, 0.35)"
            : "1px solid rgba(145, 237, 184, 0.34)",
          boxShadow: isChurn
            ? "0 14px 30px rgba(186, 34, 56, 0.35)"
            : "0 14px 30px rgba(55, 163, 95, 0.28)",
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
            bgcolor: "rgba(154, 177, 212, 0.25)",
            "& .MuiLinearProgress-bar": {
              background: isChurn
                ? "linear-gradient(90deg, #ff6f83 0%, #ff4b63 100%)"
                : "linear-gradient(90deg, #3bd37c 0%, #7fe3ab 100%)",
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
              sx={{
                fontWeight: "bold",
                fontSize: "0.9rem",
                width: "100%",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.16)",
              }}
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
              sx={{
                fontWeight: "bold",
                fontSize: "0.9rem",
                width: "100%",
                boxShadow: "0 8px 16px rgba(0, 0, 0, 0.16)",
              }}
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
