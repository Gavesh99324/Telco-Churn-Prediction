import axios, { AxiosInstance } from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export interface CustomerData {
  gender: string;
  SeniorCitizen: number;
  Partner: string;
  Dependents: string;
  tenure: number;
  PhoneService: string;
  MultipleLines: string;
  InternetService: string;
  OnlineSecurity: string;
  OnlineBackup: string;
  DeviceProtection: string;
  TechSupport: string;
  StreamingTV: string;
  StreamingMovies: string;
  Contract: string;
  PaperlessBilling: string;
  PaymentMethod: string;
  MonthlyCharges: number;
  TotalCharges: number;
}

export interface PredictionResponse {
  prediction: number;
  probability: number;
  confidence: string;
  risk_level: string;
  timestamp: string;
  model_version: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
  model_loaded: boolean;
  uptime_seconds: number;
}

export interface ModelInfo {
  model_name: string;
  model_path: string;
  model_loaded: boolean;
  feature_count: number;
  version: string;
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log(
          `API Request: ${config.method?.toUpperCase()} ${config.url}`,
        );
        return config;
      },
      (error) => {
        return Promise.reject(error);
      },
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error("API Error:", error.response?.data || error.message);
        return Promise.reject(error);
      },
    );
  }

  async health(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>("/health");
    return response.data;
  }

  async getModelInfo(): Promise<ModelInfo> {
    const response = await this.client.get<ModelInfo>("/model/info");
    return response.data;
  }

  async predictChurn(customer: CustomerData): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>(
      "/predict",
      customer,
    );
    return response.data;
  }

  async getMetrics(): Promise<string> {
    const response = await this.client.get<string>("/metrics");
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;
