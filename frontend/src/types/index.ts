export interface PredictionHistory {
  id: string;
  timestamp: string;
  customerInfo: string;
  prediction: number;
  probability: number;
  riskLevel: string;
  confidence: string;
}

export const GENDER_OPTIONS = ["Male", "Female"];
export const YES_NO_OPTIONS = ["Yes", "No"];
export const INTERNET_SERVICE_OPTIONS = ["DSL", "Fiber optic", "No"];
export const MULTIPLE_LINES_OPTIONS = ["Yes", "No", "No phone service"];
export const ONLINE_SERVICE_OPTIONS = ["Yes", "No", "No internet service"];
export const CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"];
export const PAYMENT_METHOD_OPTIONS = [
  "Electronic check",
  "Mailed check",
  "Bank transfer (automatic)",
  "Credit card (automatic)",
];

export const DEFAULT_CUSTOMER_DATA = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70.7,
  TotalCharges: 848.4,
};
