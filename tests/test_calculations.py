"""
Test suite focusing on calculation accuracy of temperature aggregation API.
"""

import pytest
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

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
    
    resampled = window_data.resample(
        f'{resolution}S',
        closed='left',
        label='left'
    ).agg({
        'ambient_temperature': ['mean', 'min', 'max'],
        'device_temperature': ['mean', 'min', 'max']
    })
    
    return resampled


def test_one_hour_mean_accuracy():
    """Test 1: Verify mean calculation accuracy for one-hour window."""
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(hours=1)
    
    api_data = get_api_data(start_time, end_time, 3600)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 3600)
    
    api_ambient_mean = api_data['aggregated_data'][0]['ambient_temperature']['mean']
    manual_ambient_mean = round(manual_stats['ambient_temperature']['mean'].iloc[0], 2)
    
    assert api_ambient_mean == manual_ambient_mean


def test_daily_min_max_accuracy():
    """Test 2: Verify min/max accuracy over 24-hour period."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=24)
    
    api_data = get_api_data(start_time, end_time, 86400)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 86400)
    
    point = api_data['aggregated_data'][0]
    assert point['ambient_temperature']['min'] == round(manual_stats['ambient_temperature']['min'].iloc[0], 2)
    assert point['ambient_temperature']['max'] == round(manual_stats['ambient_temperature']['max'].iloc[0], 2)
    assert point['device_temperature']['min'] == round(manual_stats['device_temperature']['min'].iloc[0], 2)
    assert point['device_temperature']['max'] == round(manual_stats['device_temperature']['max'].iloc[0], 2)


@pytest.mark.parametrize("resolution", [60, 300, 600, 1800, 3600])
def test_resolution_accuracy(resolution):
    """Tests 3-7: Verify calculation accuracy across different resolutions."""
    start_time = datetime(2024, 1, 1, 12, 0, 0)
    end_time = start_time + timedelta(hours=1)
    
    api_data = get_api_data(start_time, end_time, resolution)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, resolution)
    
    for api_point, (manual_idx, manual_row) in zip(api_data['aggregated_data'], manual_stats.iterrows()):
        # Check ambient temperature calculations
        assert api_point['ambient_temperature']['mean'] == round(manual_row['ambient_temperature']['mean'], 2)
        assert api_point['ambient_temperature']['min'] == round(manual_row['ambient_temperature']['min'], 2)
        assert api_point['ambient_temperature']['max'] == round(manual_row['ambient_temperature']['max'], 2)
        
        # Check device temperature calculations
        assert api_point['device_temperature']['mean'] == round(manual_row['device_temperature']['mean'], 2)
        assert api_point['device_temperature']['min'] == round(manual_row['device_temperature']['min'], 2)
        assert api_point['device_temperature']['max'] == round(manual_row['device_temperature']['max'], 2)


def test_spike_max_accuracy():
    """Test 8: Verify max calculation during known temperature spikes."""
    # Load spike times
    spike_times = pd.read_csv('spike_times.csv')
    spike_times['start_time'] = pd.to_datetime(spike_times['start_time'])
    spike_times['peak_time'] = pd.to_datetime(spike_times['peak_time'])
    
    for _, spike in spike_times.iterrows():
        start_time = spike['start_time']
        end_time = start_time + timedelta(minutes=5)
        
        api_data = get_api_data(start_time, end_time, 300)
        manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 300)
        
        api_max = api_data['aggregated_data'][0]['device_temperature']['max']
        manual_max = round(manual_stats['device_temperature']['max'].iloc[0], 2)
        
        assert api_max == manual_max
        assert api_max >= 70  # Verify spike detection


def test_ambient_mean_ranges():
    """Test 9: Verify ambient temperature mean calculations stay within expected ranges."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)
    
    api_data = get_api_data(start_time, end_time, 3600)
    
    for point in api_data['aggregated_data']:
        mean_temp = point['ambient_temperature']['mean']
        # Verify against known temperature range in data generation
        assert 20 <= mean_temp <= 25


def test_device_mean_ranges():
    """Test 10: Verify device temperature mean calculations stay within expected ranges."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)
    
    api_data = get_api_data(start_time, end_time, 3600)
    
    for point in api_data['aggregated_data']:
        mean_temp = point['device_temperature']['mean']
        # Verify against known temperature range in data generation
        assert 45 <= mean_temp <= 85


def test_temperature_relationship():
    """Test 11: Verify relationship between ambient and device temperatures."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)
    
    api_data = get_api_data(start_time, end_time, 3600)
    
    for point in api_data['aggregated_data']:
        ambient_mean = point['ambient_temperature']['mean']
        device_mean = point['device_temperature']['mean']
        # Device should always be warmer than ambient
        assert device_mean > ambient_mean
        # Verify temperature difference is within expected range
        assert 25 <= (device_mean - ambient_mean) <= 60


def test_min_max_relationship():
    """Test 12: Verify min <= mean <= max relationship."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=48)
    
    api_data = get_api_data(start_time, end_time, 3600)
    
    for point in api_data['aggregated_data']:
        # Check ambient temperature relationship
        assert point['ambient_temperature']['min'] <= point['ambient_temperature']['mean']
        assert point['ambient_temperature']['mean'] <= point['ambient_temperature']['max']
        
        # Check device temperature relationship
        assert point['device_temperature']['min'] <= point['device_temperature']['mean']
        assert point['device_temperature']['mean'] <= point['device_temperature']['max']


def test_rounding_consistency():
    """Test 13: Verify consistent decimal rounding."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=1)
    
    api_data = get_api_data(start_time, end_time, 300)
    
    for point in api_data['aggregated_data']:
        for temp_type in ['ambient_temperature', 'device_temperature']:
            for stat in ['mean', 'min', 'max']:
                value = point[temp_type][stat]
                # Check that values are rounded to 2 decimal places
                assert abs(value - round(value, 2)) < 1e-10


@pytest.mark.parametrize("window_size", [1, 6, 12, 24])
def test_calculation_window_sizes(window_size):
    """Tests 14-17: Verify calculation accuracy across different window sizes."""
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = start_time + timedelta(hours=window_size)
    
    api_data = get_api_data(start_time, end_time, 3600)
    manual_stats = calculate_manual_stats(TEST_DATA, start_time, end_time, 3600)
    
    assert len(api_data['aggregated_data']) == len(manual_stats)
    
    for api_point, (manual_idx, manual_row) in zip(api_data['aggregated_data'], manual_stats.iterrows()):
        for temp_type in ['ambient_temperature', 'device_temperature']:
            for stat in ['mean', 'min', 'max']:
                api_value = api_point[temp_type][stat]
                manual_value = round(manual_row[f'{temp_type}'][stat], 2)
                assert api_value == manual_value


def test_boundary_calculations():
    """Test 18: Verify calculations at dataset boundaries."""
    # Get first and last hour of data
    start_time = TEST_DATA.index.min()
    mid_time = start_time + timedelta(hours=1)
    end_time = TEST_DATA.index.max()
    start_end_time = end_time - timedelta(hours=1)
    
    # Test first hour
    first_hour = get_api_data(start_time, mid_time, 3600)
    first_manual = calculate_manual_stats(TEST_DATA, start_time, mid_time, 3600)
    
    # Test last hour
    last_hour = get_api_data(start_end_time, end_time, 3600)
    last_manual = calculate_manual_stats(TEST_DATA, start_end_time, end_time, 3600)
    
    # Verify calculations at both boundaries
    for api_data, manual_stats in [(first_hour, first_manual), (last_hour, last_manual)]:
        api_point = api_data['aggregated_data'][0]
        manual_row = manual_stats.iloc[0]
        
        for temp_type in ['ambient_temperature', 'device_temperature']:
            for stat in ['mean', 'min', 'max']:
                api_value = api_point[temp_type][stat]
                manual_value = round(manual_row[f'{temp_type}'][stat], 2)
                assert api_value == manual_value


def test_spike_detection_accuracy():
    """Test 19: Verify accurate detection and calculation of temperature spikes."""
    spike_times = pd.read_csv('spike_times.csv')
    spike_times['start_time'] = pd.to_datetime(spike_times['start_time'])
    
    # Check each spike
    for _, spike in spike_times.iterrows():
        window_start = spike['start_time']
        window_end = window_start + timedelta(minutes=5)
        
        api_data = get_api_data(window_start, window_end, 300)
        manual_stats = calculate_manual_stats(TEST_DATA, window_start, window_end, 300)
        
        # Verify max temperature during spike
        api_max = api_data['aggregated_data'][0]['device_temperature']['max']
        manual_max = round(manual_stats['device_temperature']['max'].iloc[0], 2)
        
        assert api_max == manual_max
        assert 70 <= api_max <= 80  # Known spike range
        assert api_max > api_data['aggregated_data'][0]['device_temperature']['mean']


def test_aggregation_count_accuracy():
    """Test 20: Verify accuracy of data point counts in aggregation."""
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
