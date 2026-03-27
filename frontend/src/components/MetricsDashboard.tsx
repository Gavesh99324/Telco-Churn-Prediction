import { useMemo } from "react";
import { Box, Grid, Paper, Typography } from "@mui/material";
import { Assessment as AssessmentIcon } from "@mui/icons-material";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { PredictionHistory } from "../types";

interface MetricsDashboardProps {
  history: PredictionHistory[];
}

export default function MetricsDashboard({ history }: MetricsDashboardProps) {
  const metrics = useMemo(() => {
    if (history.length === 0) {
      return {
        totalPredictions: 0,
        churnCount: 0,
        churnRate: 0,
        avgProbability: 0,
        highRiskCount: 0,
      };
    }

    const totalPredictions = history.length;
    const churnCount = history.filter((h) => h.prediction === 1).length;
    const churnRate = (churnCount / totalPredictions) * 100;
    const avgProbability =
      history.reduce((sum, h) => sum + h.probability, 0) / totalPredictions;
    const highRiskCount = history.filter((h) => h.riskLevel === "High").length;

    return {
      totalPredictions,
      churnCount,
      churnRate,
      avgProbability,
      highRiskCount,
    };
  }, [history]);

  // Chart data
  const timeSeriesData = useMemo(() => {
    return history
      .slice(-20)
      .reverse()
      .map((item, index) => ({
        name: `#${index + 1}`,
        probability: (item.probability * 100).toFixed(1),
      }));
  }, [history]);

  const pieData = useMemo(() => {
    const churnCount = history.filter((h) => h.prediction === 1).length;
    const noChurnCount = history.length - churnCount;

    return [
      { name: "Churn", value: churnCount, color: "#f44336" },
      { name: "No Churn", value: noChurnCount, color: "#4caf50" },
    ];
  }, [history]);

  if (history.length === 0) {
    return (
      <Box sx={{ textAlign: "center", py: 6 }}>
        <AssessmentIcon sx={{ fontSize: 60, color: "text.secondary", mb: 2 }} />
        <Typography variant="body1" color="text.secondary">
          No predictions yet. Make your first prediction to see metrics.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              textAlign: "center",
              background:
                "linear-gradient(132deg, rgba(18, 118, 186, 0.88) 0%, rgba(54, 158, 232, 0.86) 100%)",
              border: "1px solid rgba(142, 208, 255, 0.34)",
              boxShadow: "0 12px 26px rgba(13, 97, 154, 0.34)",
              color: "white",
            }}
          >
            <Typography variant="h4" fontWeight="bold">
              {metrics.totalPredictions}
            </Typography>
            <Typography variant="body2">Total Predictions</Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              textAlign: "center",
              background:
                "linear-gradient(132deg, rgba(150, 29, 47, 0.92) 0%, rgba(225, 72, 97, 0.86) 100%)",
              border: "1px solid rgba(255, 170, 182, 0.34)",
              boxShadow: "0 12px 26px rgba(171, 38, 61, 0.3)",
              color: "white",
            }}
          >
            <Typography variant="h4" fontWeight="bold">
              {metrics.churnCount}
            </Typography>
            <Typography variant="body2">Churn Predictions</Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              textAlign: "center",
              background:
                metrics.churnRate > 50
                  ? "linear-gradient(132deg, rgba(112, 25, 42, 0.92) 0%, rgba(189, 56, 76, 0.88) 100%)"
                  : "linear-gradient(132deg, rgba(24, 109, 61, 0.92) 0%, rgba(67, 182, 114, 0.88) 100%)",
              border:
                metrics.churnRate > 50
                  ? "1px solid rgba(255, 166, 181, 0.32)"
                  : "1px solid rgba(145, 241, 188, 0.34)",
              boxShadow:
                metrics.churnRate > 50
                  ? "0 12px 26px rgba(148, 36, 57, 0.3)"
                  : "0 12px 26px rgba(35, 130, 78, 0.3)",
              color: "white",
            }}
          >
            <Typography variant="h4" fontWeight="bold">
              {metrics.churnRate.toFixed(1)}%
            </Typography>
            <Typography variant="body2">Churn Rate</Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Paper
            elevation={0}
            sx={{
              p: 2,
              textAlign: "center",
              background:
                "linear-gradient(132deg, rgba(179, 107, 8, 0.9) 0%, rgba(252, 166, 44, 0.86) 100%)",
              border: "1px solid rgba(255, 215, 155, 0.34)",
              boxShadow: "0 12px 24px rgba(165, 111, 28, 0.28)",
              color: "white",
            }}
          >
            <Typography variant="h4" fontWeight="bold">
              {metrics.highRiskCount}
            </Typography>
            <Typography variant="body2">High Risk</Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Line Chart */}
        <Grid item xs={12} md={8}>
          <Paper elevation={0} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Churn Probability Trend (Last 20 Predictions)
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={timeSeriesData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="probability"
                  stroke="#1976d2"
                  strokeWidth={2}
                  name="Churn Probability (%)"
                />
              </LineChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Pie Chart */}
        <Grid item xs={12} md={4}>
          <Paper elevation={0} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Churn Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={(entry) => `${entry.name}: ${entry.value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
