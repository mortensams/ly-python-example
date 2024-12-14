"""
Test suite focusing on calculation accuracy of temperature aggregation API.
"""

from datetime import datetime, timedelta

import pytest
import requests
import pandas as pd

# Constants
BASE_URL = "http://localhost:8000"
TIMEOUT = 5

# Load the source data directly for comparison
TEST_DATA = pd.read_csv('temperature_data.csv')
TEST_DATA['timestamp'] = pd.to_datetime(TEST_DATA['timestamp'])
TEST_DATA.set_index('timestamp', inplace=True)


def get_api_data(start_time, end_time, resolution):
    """Helper function to get data from API."""
    params = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "resolution": resolution
    }
    response = requests.get(f"{BASE_URL}/aggregate", params=params, timeout=TIMEOUT)
    assert response.status_code == 200
    return response.json()


def calculate_manual_stats(data, start_time, end_time, resolution):
    """Calculate statistics manually for comparison."""
    mask = (data.index >= start_time) & (data.index < end_time)
    window_data = data.loc[mask]

    return window_data.resample(
        f'{resolution}S',
        closed='left',
        label='left'
    ).agg({
        'ambient_temperature': ['mean', 'min', 'max'],
        'device_temperature': ['mean', 'min', 'max']
    })


def get_temperature_stats(row, temp_type):
    """Helper to extract temperature statistics."""
    return {
        'mean': round(row[f'{temp_type}']['mean'], 2),
        'min': round(row[f'{temp_type}']['min'], 2),
        'max': round(row[f'{temp_type}']['max'], 2)
    }


def test_one_hour_mean_accuracy():
    """Test 1: Verify mean calculation accuracy for one-hour window."""
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(hours=1)

    api_data = get_api_data(start_time, end_time, 3600)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 3600)

    api_ambient_mean = api_data['aggregated_data'][0]['ambient_temperature']['mean']
    manual_mean = round(manual_stats['ambient_temperature']['mean'].iloc[0], 2)

    assert api_ambient_mean == manual_mean


def test_daily_min_max_accuracy():
    """Test 2: Verify min/max accuracy over 24-hour period."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=24)

    api_data = get_api_data(start_time, end_time, 86400)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 86400)

    point = api_data['aggregated_data'][0]
    row = manual_stats.iloc[0]

    for temp_type in ['ambient_temperature', 'device_temperature']:
        api_stats = point[temp_type]
        manual_stats = get_temperature_stats(row, temp_type)
        assert api_stats == manual_stats


@pytest.mark.parametrize("resolution", [60, 300, 600, 1800, 3600])
def test_resolution_accuracy(resolution):
    """Tests 3-7: Verify calculation accuracy across different resolutions."""
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(hours=1)

    api_data = get_api_data(start_time, end_time, resolution)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, resolution)

    for point, row in zip(api_data['aggregated_data'], manual_stats.itertuples()):
        for temp_type in ['ambient_temperature', 'device_temperature']:
            api_stats = point[temp_type]
            manual_vals = get_temperature_stats(row, temp_type)
            assert api_stats == manual_vals


def test_ambient_mean_ranges():
    """Test 8: Verify ambient temperature mean calculations stay within range."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)

    api_data = get_api_data(start_time, end_time, 3600)

    for point in api_data['aggregated_data']:
        mean_temp = point['ambient_temperature']['mean']
        assert 20 <= mean_temp <= 25


def test_device_mean_ranges():
    """Test 9: Verify device temperature mean calculations stay within range."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)

    api_data = get_api_data(start_time, end_time, 3600)

    for point in api_data['aggregated_data']:
        mean_temp = point['device_temperature']['mean']
        assert 45 <= mean_temp <= 85


def test_temperature_relationship():
    """Test 10: Verify relationship between ambient and device temperatures."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)

    api_data = get_api_data(start_time, end_time, 3600)

    for point in api_data['aggregated_data']:
        ambient_mean = point['ambient_temperature']['mean']
        device_mean = point['device_temperature']['mean']
        temp_diff = device_mean - ambient_mean
        assert device_mean > ambient_mean
        assert 25 <= temp_diff <= 60


def test_min_max_relationship():
    """Test 11: Verify min <= mean <= max relationship."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)

    api_data = get_api_data(start_time, end_time, 3600)

    for point in api_data['aggregated_data']:
        ambient = point['ambient_temperature']
        device = point['device_temperature']

        assert ambient['min'] <= ambient['mean'] <= ambient['max']
        assert device['min'] <= device['mean'] <= device['max']


def test_rounding_consistency():
    """Test 12: Verify consistent decimal rounding."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=1)

    api_data = get_api_data(start_time, end_time, 300)

    for point in api_data['aggregated_data']:
        for temp_type in ['ambient_temperature', 'device_temperature']:
            for stat in ['mean', 'min', 'max']:
                value = point[temp_type][stat]
                assert abs(value - round(value, 2)) < 1e-10


@pytest.mark.parametrize("window_size", [1, 6, 12, 24])
def test_calculation_window_sizes(window_size):
    """Tests 13-16: Verify calculation accuracy across different window sizes."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=window_size)

    api_data = get_api_data(start_time, end_time, 3600)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 3600)

    assert len(api_data['aggregated_data']) == len(manual_stats)

    for point, row in zip(api_data['aggregated_data'], manual_stats.itertuples()):
        for temp_type in ['ambient_temperature', 'device_temperature']:
            api_stats = point[temp_type]
            manual_stats = get_temperature_stats(row, temp_type)
            assert api_stats == manual_stats


def verify_boundary_stats(api_data, manual_stats):
    """Helper for boundary tests."""
    point = api_data['aggregated_data'][0]
    row = manual_stats.iloc[0]

    for temp_type in ['ambient_temperature', 'device_temperature']:
        api_stats = point[temp_type]
        manual_vals = get_temperature_stats(row, temp_type)
        assert api_stats == manual_vals


def test_boundary_calculations():
    """Test 17: Verify calculations at dataset boundaries."""
    start_time = TEST_DATA.index.min()
    end_time = TEST_DATA.index.max()
    hour_delta = timedelta(hours=1)

    # Test first hour
    first_hour = get_api_data(start_time, start_time + hour_delta, 3600)
    first_stats = calculate_manual_stats(TEST_DATA, start_time, start_time + hour_delta, 3600)
    verify_boundary_stats(first_hour, first_stats)

    # Test last hour
    last_start = end_time - hour_delta
    last_hour = get_api_data(last_start, end_time, 3600)
    last_stats = calculate_manual_stats(TEST_DATA, last_start, end_time, 3600)
    verify_boundary_stats(last_hour, last_stats)


def test_aggregation_count_accuracy():
    """Test 18: Verify accuracy of data point counts in aggregation."""
    test_cases = [
        (60, 60),    # 1 hour at 1-minute resolution
        (300, 12),   # 1 hour at 5-minute resolution
        (600, 6),    # 1 hour at 10-minute resolution
        (1800, 2),   # 1 hour at 30-minute resolution
        (3600, 1)    # 1 hour at 1-hour resolution
    ]

    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(hours=1)

    for resolution, expected_count in test_cases:
        api_data = get_api_data(start_time, end_time, resolution)
        assert len(api_data['aggregated_data']) == expected_count
        assert api_data['data_points'] == expected_count