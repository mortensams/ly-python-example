from fastapi import FastAPI, Query, HTTPException
from datetime import datetime
import pandas as pd
from typing import Optional
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

@app.get("/aggregate")
async def aggregate_temperatures(
    start_time: datetime = Query(
        ...,
        description="Start time (ISO 8601 format, e.g., 2024-01-01T12:00:00)"
    ),
    end_time: datetime = Query(
        ...,
        description="End time (ISO 8601 format, e.g., 2024-01-01T15:00:00)"
    ),
    resolution: int = Query(
        default=60,
        gt=0,  # Must be greater than 0
        description="Aggregation resolution in seconds"
    )
):
    """
    Aggregate temperature data for a given time window and resolution.
    """
    if df is None:
        raise HTTPException(status_code=500, detail="Data file not loaded")
    
    # Validate time range
    if end_time <= start_time:
        raise HTTPException(
            status_code=400,
            detail="End time must be after start time"
        )

    # Make end_time exclusive to fix off-by-one issues
    end_time_exclusive = end_time
    start_time_inclusive = start_time

    # Filter data for the requested time window
    mask = (df.index >= start_time_inclusive) & (df.index < end_time_exclusive)
    window_data = df.loc[mask]
    
    if window_data.empty:
        raise HTTPException(
            status_code=404,
            detail="No data found for the specified time range"
        )

    # Resample and aggregate data
    resampled = window_data.resample(f'{resolution}S', closed='left', label='left').agg({
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
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'data_points': len(result),
        'aggregated_data': result
    }

@app.get("/health")
async def health_check():
    """Check if the service is healthy and data is loaded"""
    return {
        "status": "healthy" if df is not None else "unhealthy",
        "data_loaded": df is not None,
        "total_records": len(df) if df is not None else 0,
        "time_range": {
            "start": df.index.min().isoformat() if df is not None else None,
            "end": df.index.max().isoformat() if df is not None else None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)