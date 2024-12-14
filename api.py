"""
FastAPI service for aggregating and analyzing temperature time series data.
Provides endpoints for data aggregation with flexible time windows and resolutions.
"""

import re
from datetime import datetime
from typing import List, Dict
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Temperature Analysis API",
    description="Temperature data analysis and aggregation capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the data once when starting the service
try:
    DATA_FRAME = pd.read_csv('temperature_data.csv')
    DATA_FRAME['timestamp'] = pd.to_datetime(DATA_FRAME['timestamp']).dt.tz_localize('UTC')
    DATA_FRAME.set_index('timestamp', inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    DATA_FRAME = None

class TemperatureStats(BaseModel):
    """Statistics for temperature measurements."""
    mean: float = Field(description="Average temperature")
    min: float = Field(description="Minimum temperature")
    max: float = Field(description="Maximum temperature")

class AggregatedDataPoint(BaseModel):
    """Single data point with temperature statistics."""
    timestamp: str = Field(description="ISO 8601 timestamp")
    ambient_temperature: TemperatureStats
    device_temperature: TemperatureStats

class AggregationResponse(BaseModel):
    """Response model for temperature aggregation."""
    resolution_seconds: int = Field(description="Time bucket size in seconds")
    start_time: str = Field(description="Start time (ISO 8601)")
    end_time: str = Field(description="End time (ISO 8601)")
    data_points: int = Field(description="Number of data points")
    aggregated_data: List[AggregatedDataPoint]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service health status")
    data_loaded: bool = Field(description="Data loading status")
    total_records: int = Field(description="Total number of records")
    time_range: Dict[str, str] = Field(description="Available data time range")

@app.get("/health", response_model=HealthResponse)
@app.head("/health")  # Allow HEAD requests for healthcheck
async def health_check() -> HealthResponse:
    """Check if the service is healthy and data is loaded."""
    if DATA_FRAME is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unhealthy - data not loaded"
        )

    return {
        "status": "healthy",
        "data_loaded": True,
        "total_records": len(DATA_FRAME),
        "time_range": {
            "start": DATA_FRAME.index.min().isoformat(),
            "end": DATA_FRAME.index.max().isoformat()
        }
    }

@app.get(
    "/aggregate",
    response_model=AggregationResponse,
    responses={
        status.HTTP_200_OK: {"description": "Successfully aggregated temperature data"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid request parameters"},
        status.HTTP_404_NOT_FOUND: {"description": "No data found"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Validation error"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Server error"}
    }
)
async def aggregate_temperatures(
    start_time: str = Query(..., description="Start time (ISO 8601)"),
    end_time: str = Query(..., description="End time (ISO 8601)"),
    resolution: int = Query(60, gt=0, le=86400, description="Aggregation resolution in seconds")
) -> AggregationResponse:
    """Aggregate temperature data for a given time window and resolution."""
    try:
        start_dt = pd.to_datetime(start_time).tz_localize('UTC')
        end_dt = pd.to_datetime(end_time).tz_localize('UTC')
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        ) from e

    if DATA_FRAME is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data file not loaded"
        )

    # Validate time range
    if end_dt <= start_dt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="End time must be after start time"
        )

    # Filter data for the requested time window
    mask = (DATA_FRAME.index >= start_dt) & (DATA_FRAME.index < end_dt)
    window_data = DATA_FRAME.loc[mask]

    if window_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data found for the specified time range"
        )

    # Resample and aggregate data
    resampled = window_data.resample(
        f'{resolution}S',
        closed='left',
        label='left'
    ).agg({
        'ambient_temperature': ['mean', 'min', 'max'],
        'device_temperature': ['mean', 'min', 'max']
    })

    # Flatten column names
    resampled.columns = [f"{col[0]}_{col[1]}" for col in resampled.columns]

    # Convert timestamps to ISO format and prepare response
    result = []
    for timestamp, row in resampled.iterrows():
        result.append({
            'timestamp': timestamp.isoformat(),
            'ambient_temperature': {
                'mean': round(row['ambient_temperature_mean'], 2),
                'min': round(row['ambient_temperature_min'], 2),
                'max': round(row['ambient_temperature_max'], 2)
            },
            'device_temperature': {
                'mean': round(row['device_temperature_mean'], 2),
                'min': round(row['device_temperature_min'], 2),
                'max': round(row['device_temperature_max'], 2)
            }
        })

    return {
        'resolution_seconds': resolution,
        'start_time': start_dt.isoformat(),
        'end_time': end_dt.isoformat(),
        'data_points': len(result),
        'aggregated_data': result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)