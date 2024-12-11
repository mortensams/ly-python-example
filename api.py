"""
FastAPI service for aggregating and analyzing temperature time series data.
Provides endpoints for data aggregation with flexible time windows and resolutions.
"""

import re
from datetime import datetime
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, status
from pydantic import BaseModel, Field

# Service metadata
DESCRIPTION = """
This API provides temperature data analysis and aggregation capabilities.
It handles both ambient and device temperatures, allowing for flexible time windows
and aggregation resolutions.

Key features:
* Time-based data aggregation
* Configurable resolution (time buckets)
* Statistical analysis (min, max, mean)
* Health monitoring
"""

app = FastAPI(
    title="Temperature Analysis API",
    description=DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Your Name",
        "url": "https://github.com/yourusername/ly-python-example",
        "email": "your.email@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)


def parse_datetime(value: str) -> datetime:
    """
    Parse datetime string in various ISO 8601 formats and convert to UTC.
    
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
    mean: float = Field(description="Average temperature for the time window")
    min: float = Field(description="Minimum temperature in the time window")
    max: float = Field(description="Maximum temperature in the time window")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "mean": 22.5,
                "min": 21.0,
                "max": 24.0
            }]
        }
    }


class AggregatedDataPoint(BaseModel):
    """Single data point containing timestamp and temperature statistics."""
    timestamp: str = Field(description="ISO 8601 formatted timestamp")
    ambient_temperature: TemperatureStats = Field(description="Ambient temperature statistics")
    device_temperature: TemperatureStats = Field(description="Device temperature statistics")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "timestamp": "2024-01-01T12:00:00",
                "ambient_temperature": {
                    "mean": 22.5,
                    "min": 21.0,
                    "max": 24.0
                },
                "device_temperature": {
                    "mean": 55.5,
                    "min": 53.0,
                    "max": 58.0
                }
            }]
        }
    }


class AggregationResponse(BaseModel):
    """Response model for temperature aggregation endpoint."""
    resolution_seconds: int = Field(description="Time bucket size in seconds")
    start_time: str = Field(description="Start of the aggregation window (ISO 8601)")
    end_time: str = Field(description="End of the aggregation window (ISO 8601)")
    data_points: int = Field(description="Number of data points in response")
    aggregated_data: List[AggregatedDataPoint] = Field(description="Aggregated temperature data")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "resolution_seconds": 300,
                "start_time": "2024-01-01T12:00:00",
                "end_time": "2024-01-01T13:00:00",
                "data_points": 12,
                "aggregated_data": [
                    {
                        "timestamp": "2024-01-01T12:00:00",
                        "ambient_temperature": {
                            "mean": 22.5,
                            "min": 21.0,
                            "max": 24.0
                        },
                        "device_temperature": {
                            "mean": 55.5,
                            "min": 53.0,
                            "max": 58.0
                        }
                    }
                ]
            }]
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(description="Service health status (healthy/unhealthy)")
    data_loaded: bool = Field(description="Whether temperature data is loaded")
    total_records: int = Field(description="Total number of temperature records")
    time_range: Dict[str, str] = Field(description="Available data time range")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "status": "healthy",
                "data_loaded": True,
                "total_records": 172800,
                "time_range": {
                    "start": "2024-01-01T00:00:00",
                    "end": "2024-01-02T23:59:59"
                }
            }]
        }
    }


@app.get(
    "/aggregate",
    response_model=AggregationResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Successfully aggregated temperature data",
            "model": AggregationResponse
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_time_range": {
                            "summary": "Invalid time range",
                            "value": {"detail": "End time must be after start time"}
                        }
                    }
                }
            }
        },
        status.HTTP_404_NOT_FOUND: {
            "description": "No data found",
            "content": {
                "application/json": {
                    "examples": {
                        "no_data": {
                            "summary": "No data in range",
                            "value": {"detail": "No data found for the specified time range"}
                        }
                    }
                }
            }
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "description": "Validation error",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_format": {
                            "summary": "Invalid datetime format",
                            "value": {"detail": "Invalid datetime format"}
                        }
                    }
                }
            }
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Server error",
            "content": {
                "application/json": {
                    "examples": {
                        "data_not_loaded": {
                            "summary": "Data not loaded",
                            "value": {"detail": "Data file not loaded"}
                        }
                    }
                }
            }
        }
    },
    summary="Aggregate temperature data",
    description="""
    Aggregates temperature data over a specified time window with given resolution.
    
    The endpoint provides:
    * Time-based aggregation of temperature data
    * Configurable resolution (time bucket size)
    * Statistical analysis (min, max, mean) for each bucket
    * Both ambient and device temperature analysis
    
    Example usage:
    ```
    /aggregate?start_time=2024-01-01T12:00:00&end_time=2024-01-01T13:00:00&resolution=300
    ```
    This will return 5-minute (300 seconds) aggregated data for one hour.
    """
)
async def aggregate_temperatures(
    start_time: str = Query(
        description="Start time in ISO 8601 format",
        examples=["2024-01-01T12:00:00"]
    ),
    end_time: str = Query(
        description="End time in ISO 8601 format",
        examples=["2024-01-01T13:00:00"]
    ),
    resolution: int = Query(
        default=60,
        gt=0,
        le=86400,  # Max 1 day resolution
        description="Aggregation resolution in seconds (1 to 86400)",
        examples=[300]
    )
) -> AggregationResponse:
    """
    Aggregate temperature data for a given time window and resolution.
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
        status.HTTP_200_OK: {
            "description": "Service is healthy",
            "model": HealthResponse
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Service is unhealthy",
            "content": {
                "application/json": {
                    "examples": {
                        "unhealthy": {
                            "summary": "Data not loaded",
                            "value": {"detail": "Service unhealthy - data not loaded"}
                        }
                    }
                }
            }
        }
    },
    summary="Check service health",
    description="""
    Provides health status of the service including:
    * Service status (healthy/unhealthy)
    * Data loading status
    * Total number of records
    * Available data time range
    """
)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
