"""
Test suite for temperature aggregation API endpoints.
Tests data aggregation, validation, and error handling.
"""

import time
import pytest
import requests


BASE_URL = "http://localhost:8000"
TIMEOUT = 5  # seconds


def wait_for_service():
    """Wait for the service to become available."""
    max_attempts = 30
    for _ in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_service():
    """Ensure service is running before tests."""
    assert wait_for_service(), "Service failed to start"


def test_health_check():
    """Test 1: Health check endpoint."""
    response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_basic_aggregation():
    """Test 2: Basic aggregation with default resolution."""
    params = {
        "start_time": "2024-01-01T12:00:00",
        "end_time": "2024-01-01T13:00:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 200
    data = response.json()
    assert len(data["aggregated_data"]) == 60  # 1 hour with 1-minute resolution


def test_custom_resolution():
    """Test 3: Custom resolution of 5 minutes."""
    params = {
        "start_time": "2024-01-01T12:00:00",
        "end_time": "2024-01-01T13:00:00",
        "resolution": 300
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 200
    data = response.json()
    assert len(data["aggregated_data"]) == 12  # 1 hour with 5-minute resolution


def test_small_time_window():
    """Test 4: Small time window of 5 minutes."""
    params = {
        "start_time": "2024-01-01T12:00:00",
        "end_time": "2024-01-01T12:05:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 200
    data = response.json()
    assert len(data["aggregated_data"]) == 5


def test_full_day_aggregation():
    """Test 5: Full day aggregation."""
    params = {
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-02T00:00:00",
        "resolution": 3600
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 200
    data = response.json()
    assert len(data["aggregated_data"]) == 24


def test_temperature_ranges():
    """Test 6: Check temperature ranges."""
    params = {
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T01:00:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    data = response.json()
    for point in data["aggregated_data"]:
        assert 15 <= point["ambient_temperature"]["mean"] <= 30
        assert 45 <= point["device_temperature"]["mean"] <= 85


def test_temperature_stats():
    """Test 7: Check temperature statistics."""
    params = {
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T01:00:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    data = response.json()
    for point in data["aggregated_data"]:
        assert point["ambient_temperature"]["min"] <= point["ambient_temperature"]["mean"]
        assert point["ambient_temperature"]["mean"] <= point["ambient_temperature"]["max"]
        assert point["device_temperature"]["min"] <= point["device_temperature"]["mean"]
        assert point["device_temperature"]["mean"] <= point["device_temperature"]["max"]


def test_resolution_boundaries():
    """Tests 8-14: Various resolution values."""
    resolutions = [1, 30, 60, 300, 600, 1800, 3600]
    for resolution in resolutions:
        params = {
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T01:00:00",
            "resolution": resolution
        }
        response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
        assert response.status_code == 200
        data = response.json()
        assert len(data["aggregated_data"]) == 3600 // resolution


def test_end_before_start():
    """Test N2: End time before start time."""
    params = {
        "start_time": "2024-01-01T13:00:00",
        "end_time": "2024-01-01T12:00:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 400


def test_future_date():
    """Test N4: Future date range."""
    params = {
        "start_time": "2025-01-01T12:00:00",
        "end_time": "2025-01-01T13:00:00"
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 404


def test_missing_parameters():
    """Test N5: Missing required parameters."""
    response = requests.get(f"{BASE_URL}/aggregate", timeout=TIMEOUT)
    assert response.status_code == 422
