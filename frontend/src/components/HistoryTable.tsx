import { DataGrid } from "@mui/x-data-grid";
import { Box, Chip, Typography } from "@mui/material";
import { PredictionHistory } from "../types";

interface HistoryTableProps {
  history: PredictionHistory[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type CellParams = { value: any; row: any };

export default function HistoryTable({ history }: HistoryTableProps) {
  const columns = [
    {
      field: "id",
      headerName: "ID",
      width: 100,
      renderCell: (params: CellParams) => `#${params.row.id.slice(-6)}`,
    },
    {
      field: "timestamp",
      headerName: "Timestamp",
      width: 180,
      renderCell: (params: CellParams) =>
        new Date(params.value).toLocaleString(),
    },
    {
      field: "customerInfo",
      headerName: "Customer Info",
      width: 300,
      flex: 1,
    },
    {
      field: "prediction",
      headerName: "Prediction",
      width: 130,
      renderCell: (params: CellParams) => (
        <Chip
          label={params.value === 1 ? "Churn" : "No Churn"}
          color={params.value === 1 ? "error" : "success"}
          size="small"
          sx={{ fontWeight: "bold" }}
        />
      ),
    },
    {
      field: "probability",
      headerName: "Probability",
      width: 120,
      renderCell: (params: CellParams) => `${(params.value * 100).toFixed(1)}%`,
    },
    {
      field: "riskLevel",
      headerName: "Risk Level",
      width: 130,
      renderCell: (params: CellParams) => {
        const color: "error" | "warning" | "success" =
          params.value === "High"
            ? "error"
            : params.value === "Medium"
              ? "warning"
              : "success";
        return (
          <Chip
            label={params.value}
            color={color}
            size="small"
            variant="outlined"
          />
        );
      },
    },
    {
      field: "confidence",
      headerName: "Confidence",
      width: 130,
      renderCell: (params: CellParams) => {
        const color: "success" | "warning" | "error" =
          params.value === "High"
            ? "success"
            : params.value === "Medium"
              ? "warning"
              : "error";
        return (
          <Chip
            label={params.value}
            color={color}
            size="small"
            variant="outlined"
          />
        );
      },
    },
  ];

  if (history.length === 0) {
    return (
      <Box sx={{ textAlign: "center", py: 6 }}>
        <Typography variant="body1" color="text.secondary">
          No prediction history yet. Make your first prediction to see results
          here.
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ height: 400, width: "100%" }}>
      <DataGrid
        rows={history}
        columns={columns}
        initialState={{
          pagination: {
            paginationModel: { page: 0, pageSize: 10 },
          },
        }}
        pageSizeOptions={[5, 10, 25, 50]}
        disableRowSelectionOnClick
        sx={{
          border: "1px solid rgba(113, 151, 218, 0.28)",
          borderRadius: 2,
          backgroundColor: "rgba(7, 20, 40, 0.52)",
          backdropFilter: "blur(6px)",
          "& .MuiDataGrid-columnHeaders": {
            backgroundColor: "rgba(16, 38, 74, 0.72)",
            borderBottom: "1px solid rgba(113, 151, 218, 0.24)",
          },
          "& .MuiDataGrid-row": {
            borderBottom: "1px solid rgba(113, 151, 218, 0.14)",
          },
          "& .MuiDataGrid-row:hover": {
            backgroundColor: "rgba(35, 89, 164, 0.2)",
          },
          "& .MuiDataGrid-footerContainer": {
            borderTop: "1px solid rgba(113, 151, 218, 0.24)",
          },
          "& .MuiDataGrid-cell:focus": {
            outline: "none",
          },
        }}
      />
    </Box>
  );
}
