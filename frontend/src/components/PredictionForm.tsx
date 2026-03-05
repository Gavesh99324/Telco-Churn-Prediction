import { useState } from "react";
import {
  Box,
  TextField,
  MenuItem,
  Button,
  Grid,
  FormControl,
  InputLabel,
  Select,
  FormControlLabel,
  Switch,
  Divider,
  Typography,
  CircularProgress,
} from "@mui/material";
import { Send as SendIcon } from "@mui/icons-material";
import {
  GENDER_OPTIONS,
  YES_NO_OPTIONS,
  INTERNET_SERVICE_OPTIONS,
  MULTIPLE_LINES_OPTIONS,
  ONLINE_SERVICE_OPTIONS,
  CONTRACT_OPTIONS,
  PAYMENT_METHOD_OPTIONS,
  DEFAULT_CUSTOMER_DATA,
} from "../types";

interface PredictionFormProps {
  onPredict: (data: any) => void;
  loading: boolean;
  disabled?: boolean;
}

export default function PredictionForm({
  onPredict,
  loading,
  disabled,
}: PredictionFormProps) {
  const [formData, setFormData] = useState(DEFAULT_CUSTOMER_DATA);

  const handleChange = (field: string, value: any) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formData);
  };

  const handleReset = () => {
    setFormData(DEFAULT_CUSTOMER_DATA);
  };

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Grid container spacing={2}>
        {/* Demographics */}
        <Grid item xs={12}>
          <Typography variant="subtitle2" color="primary" gutterBottom>
            Demographics
          </Typography>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Gender</InputLabel>
            <Select
              value={formData.gender}
              label="Gender"
              onChange={(e) => handleChange("gender", e.target.value)}
            >
              {GENDER_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControlLabel
            control={
              <Switch
                checked={formData.SeniorCitizen === 1}
                onChange={(e) =>
                  handleChange("SeniorCitizen", e.target.checked ? 1 : 0)
                }
              />
            }
            label="Senior Citizen"
          />
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Partner</InputLabel>
            <Select
              value={formData.Partner}
              label="Partner"
              onChange={(e) => handleChange("Partner", e.target.value)}
            >
              {YES_NO_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Dependents</InputLabel>
            <Select
              value={formData.Dependents}
              label="Dependents"
              onChange={(e) => handleChange("Dependents", e.target.value)}
            >
              {YES_NO_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12}>
          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" color="primary" gutterBottom>
            Account Information
          </Typography>
        </Grid>

        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            size="small"
            type="number"
            label="Tenure (months)"
            value={formData.tenure}
            onChange={(e) =>
              handleChange("tenure", parseInt(e.target.value) || 0)
            }
            inputProps={{ min: 0, max: 100 }}
          />
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Contract</InputLabel>
            <Select
              value={formData.Contract}
              label="Contract"
              onChange={(e) => handleChange("Contract", e.target.value)}
            >
              {CONTRACT_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            size="small"
            type="number"
            label="Monthly Charges ($)"
            value={formData.MonthlyCharges}
            onChange={(e) =>
              handleChange("MonthlyCharges", parseFloat(e.target.value) || 0)
            }
            inputProps={{ min: 0, step: 0.01 }}
          />
        </Grid>

        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            size="small"
            type="number"
            label="Total Charges ($)"
            value={formData.TotalCharges}
            onChange={(e) =>
              handleChange("TotalCharges", parseFloat(e.target.value) || 0)
            }
            inputProps={{ min: 0, step: 0.01 }}
          />
        </Grid>

        <Grid item xs={12}>
          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" color="primary" gutterBottom>
            Services
          </Typography>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Phone Service</InputLabel>
            <Select
              value={formData.PhoneService}
              label="Phone Service"
              onChange={(e) => handleChange("PhoneService", e.target.value)}
            >
              {YES_NO_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Multiple Lines</InputLabel>
            <Select
              value={formData.MultipleLines}
              label="Multiple Lines"
              onChange={(e) => handleChange("MultipleLines", e.target.value)}
            >
              {MULTIPLE_LINES_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Internet Service</InputLabel>
            <Select
              value={formData.InternetService}
              label="Internet Service"
              onChange={(e) => handleChange("InternetService", e.target.value)}
            >
              {INTERNET_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Online Security</InputLabel>
            <Select
              value={formData.OnlineSecurity}
              label="Online Security"
              onChange={(e) => handleChange("OnlineSecurity", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Online Backup</InputLabel>
            <Select
              value={formData.OnlineBackup}
              label="Online Backup"
              onChange={(e) => handleChange("OnlineBackup", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Device Protection</InputLabel>
            <Select
              value={formData.DeviceProtection}
              label="Device Protection"
              onChange={(e) => handleChange("DeviceProtection", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Tech Support</InputLabel>
            <Select
              value={formData.TechSupport}
              label="Tech Support"
              onChange={(e) => handleChange("TechSupport", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Streaming TV</InputLabel>
            <Select
              value={formData.StreamingTV}
              label="Streaming TV"
              onChange={(e) => handleChange("StreamingTV", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Streaming Movies</InputLabel>
            <Select
              value={formData.StreamingMovies}
              label="Streaming Movies"
              onChange={(e) => handleChange("StreamingMovies", e.target.value)}
            >
              {ONLINE_SERVICE_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Paperless Billing</InputLabel>
            <Select
              value={formData.PaperlessBilling}
              label="Paperless Billing"
              onChange={(e) => handleChange("PaperlessBilling", e.target.value)}
            >
              {YES_NO_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        <Grid item xs={12} sm={6}>
          <FormControl fullWidth size="small">
            <InputLabel>Payment Method</InputLabel>
            <Select
              value={formData.PaymentMethod}
              label="Payment Method"
              onChange={(e) => handleChange("PaymentMethod", e.target.value)}
            >
              {PAYMENT_METHOD_OPTIONS.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Action Buttons */}
        <Grid item xs={12}>
          <Divider sx={{ my: 2 }} />
        </Grid>

        <Grid item xs={6}>
          <Button
            fullWidth
            variant="outlined"
            onClick={handleReset}
            disabled={loading || disabled}
          >
            Reset
          </Button>
        </Grid>

        <Grid item xs={6}>
          <Button
            fullWidth
            variant="contained"
            type="submit"
            disabled={loading || disabled}
            startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
          >
            {loading ? "Predicting..." : "Predict Churn"}
          </Button>
        </Grid>
      </Grid>
    </Box>
  );
}
