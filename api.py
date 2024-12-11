"""
FastAPI service for aggregating and analyzing temperature time series data.
Provides endpoints for data aggregation with flexible time windows and resolutions.
"""

import re
from datetime import datetime
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, status
from pydantic import BaseModel

app = FastAPI(title="Temperature Aggregation API")

def parse_datetime(value: str) -> datetime:
    """
    Parse datetime string in various ISO 8601 formats and convert to UTC.
    Handles basic ISO format, UTC Z suffix, and explicit timezone.

    Args:
        value: Datetime string in ISO 8601 format

    Returns:
        datetime: Parsed datetime object

    Raises:
        ValueError: If datetime format is invalid
    """
    try:
        # Handle basic format
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$', value):
            return datetime.fromisoformat(value)

        # Handle Z suffix
        if value.endswith('Z'):
            return datetime.fromisoformat(value.replace('Z', '+00:00'))

        # Handle explicit timezone
        if re.match(r'.*[+-]\d{2}:\d{2}$', value):
            dt = datetime.fromisoformat(value)
            return dt.astimezone(ZoneInfo('UTC'))

        raise ValueError("Invalid datetime format")
    except Exception as e:
        raise ValueError(f"Invalid datetime format: {str(e)}") from e

# Load the data once when starting the service
try:
    DATA_FRAME = pd.read_csv('temperature_data.csv')
    DATA_FRAME['timestamp'] = pd.to_datetime(DATA_FRAME['timestamp'])
    DATA_FRAME.set_index('timestamp', inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    DATA_FRAME = None

class TemperatureStats(BaseModel):
    """Statistics for temperature measurements including mean, min, and max values."""
    mean: float
    min: float
    max: float

class AggregatedDataPoint(BaseModel):
    """Single data point containing timestamp and temperature statistics."""
    timestamp: str
    ambient_temperature: TemperatureStats
    device_temperature: TemperatureStats

class AggregationResponse(BaseModel):
    """Response model for temperature aggregation endpoint."""
    resolution_seconds: int
    start_time: str
    end_time: str
    data_points: int
    aggregated_data: List[AggregatedDataPoint]

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    data_loaded: bool
    total_records: int
    time_range: dict

@app.get(
    "/aggregate",
    response_model=AggregationResponse,
    responses={
        status.HTTP_200_OK: {"description": "Successful aggregation"},
        status.HTTP_400_BAD_REQUEST: {"description": "Invalid time range"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Invalid parameters"},
        status.HTTP_404_NOT_FOUND: {"description": "No data found"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Server error"}
    }
)
async def aggregate_temperatures(
    start_time: str = Query(
        ...,
        description="Start time (ISO 8601 format)",
        example="2024-01-01T12:00:00"
    ),
    end_time: str = Query(
        ...,
        description="End time (ISO 8601 format)",
        example="2024-01-01T15:00:00"
    ),
    resolution: int = Query(
        default=60,
        gt=0,
        le=86400,  # Max 1 day resolution
        description="Aggregation resolution in seconds (1 to 86400)",
        example=60
    )
):
    """
    Aggregate temperature data for a given time window and resolution.
    
    Args:
        start_time: Start of the time window in ISO format
        end_time: End of the time window in ISO format
        resolution: Time bucket size in seconds

    Returns:
        Dict containing aggregated temperature data
    """
    try:
        start_dt = parse_datetime(start_time)
        end_dt = parse_datetime(end_time)
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

    # Validate resolution with time window
    time_diff = end_dt - start_dt
    if time_diff.total_seconds() < resolution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolution cannot be larger than the time window"
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
    resampled.columns = [
        f"{col[0]}_{col[1]}" for col in resampled.columns
    ]

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

@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        status.HTTP_200_OK: {"description": "Service health information"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Service unhealthy"}
    }
)
async def health_check():
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
