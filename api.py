from fastapi import FastAPI, Query, HTTPException, status
from datetime import datetime
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta

app = FastAPI(title="Temperature Aggregation API")

# Load the data once when starting the service
try:
    df = pd.read_csv('temperature_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
except Exception as e:
    print(f"Error loading data: {e}")
    df = None

class TemperatureStats(BaseModel):
    mean: float
    min: float
    max: float

class AggregatedDataPoint(BaseModel):
    timestamp: str
    ambient_temperature: TemperatureStats
    device_temperature: TemperatureStats

class AggregationResponse(BaseModel):
    resolution_seconds: int
    start_time: str
    end_time: str
    data_points: int
    aggregated_data: List[AggregatedDataPoint]

class HealthResponse(BaseModel):
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
    start_time: datetime = Query(
        ...,
        description="Start time (ISO 8601 format, e.g., 2024-01-01T12:00:00)",
        example="2024-01-01T12:00:00"
    ),
    end_time: datetime = Query(
        ...,
        description="End time (ISO 8601 format, e.g., 2024-01-01T15:00:00)",
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
    """
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data file not loaded"
        )
    
    # Validate time range
    if end_time <= start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="End time must be after start time"
        )

    # Validate resolution with time window
    time_diff = end_time - start_time
    if time_diff.total_seconds() < resolution:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resolution cannot be larger than the time window"
        )

    # Check if time range is within data bounds
    data_start = df.index.min()
    data_end = df.index.max()
    if start_time < data_start or end_time > data_end:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data only available from {data_start.isoformat()} to {data_end.isoformat()}"
        )

    # Filter data for the requested time window
    mask = (df.index >= start_time) & (df.index < end_time)
    window_data = df.loc[mask]
    
    if window_data.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data found for the specified time range"
        )

    # Resample and aggregate data
    try:
        resampled = window_data.resample(
            f'{resolution}S',
            closed='left',
            label='left'
        ).agg({
            'ambient_temperature': ['mean', 'min', 'max'],
            'device_temperature': ['mean', 'min', 'max']
        })
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during data aggregation: {str(e)}"
        )

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
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
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
    """Check if the service is healthy and data is loaded"""
    if df is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service unhealthy - data not loaded"
        )

    return {
        "status": "healthy",
        "data_loaded": True,
        "total_records": len(df),
        "time_range": {
            "start": df.index.min().isoformat(),
            "end": df.index.max().isoformat()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)