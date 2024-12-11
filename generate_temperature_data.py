"""
Generate synthetic temperature data for testing.
Creates a dataset with ambient and device temperatures, including overheating spikes.
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Set random seed for reproducibility
np.random.seed(42)

# Constants for data generation
START_TIME = datetime(2024, 1, 1, 0, 0, 0)
TOTAL_SECONDS = 172800  # 48 hours * 3600 seconds
BASE_TEMP = 22  # Base indoor temperature
DAILY_VARIATION = 1.5  # Daily temperature variation amplitude
NOISE_AMPLITUDE = 0.2  # Random noise amplitude

# Generate timestamps for 48 hours with 1-second intervals
timestamps = [START_TIME + timedelta(seconds=x) for x in range(TOTAL_SECONDS)]

# Convert time to hours for temperature calculation
time_hours = np.arange(TOTAL_SECONDS) / 3600

# Create smooth daily temperature cycle with noise
ambient_temp = (
    BASE_TEMP +
    DAILY_VARIATION * np.sin(2 * np.pi * time_hours / 24) +
    NOISE_AMPLITUDE * np.random.randn(TOTAL_SECONDS)
)

# Constants for device temperature
BASE_DEVICE_TEMP = 55  # Base device temperature
DEVICE_TEMP_FACTOR = 2  # How much ambient temperature changes affect device
DEVICE_NOISE = 0.5  # Additional noise for device temperature

# Generate device temperature that correlates with ambient temperature
device_temp = (
    BASE_DEVICE_TEMP +
    DEVICE_TEMP_FACTOR * (ambient_temp - BASE_TEMP) +
    DEVICE_NOISE * np.random.randn(TOTAL_SECONDS)
)

# Constants for temperature spikes
NUM_SPIKES = 8
SPIKE_DURATION = 300  # 5 minutes in seconds
MIN_GAP = 3600  # Minimum 1 hour between spikes

# Generate random spike start times (ensuring they don't overlap)
spike_starts = []
while len(spike_starts) < NUM_SPIKES:
    new_start = np.random.randint(0, TOTAL_SECONDS - SPIKE_DURATION)
    # Check if new spike is far enough from existing spikes
    if not any(abs(new_start - existing) < MIN_GAP for existing in spike_starts):
        spike_starts.append(new_start)

# Sort spike starts for better readability in data
spike_starts.sort()

# Add each spike to the device temperature
for start in spike_starts:
    # Generate a random peak temperature between 70 and 80 degrees
    peak_temp = np.random.uniform(70, 80)

    # Create a smooth spike using a half sine wave
    for i in range(SPIKE_DURATION):
        position = i / SPIKE_DURATION
        # Smooth ramp up and down using sine wave
        spike_multiplier = np.sin(position * np.pi)
        # Calculate temperature difference from current to peak
        temp_diff = peak_temp - device_temp[start + i]
        # Add the spike
        device_temp[start + i] += temp_diff * spike_multiplier

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'ambient_temperature': ambient_temp,
    'device_temperature': device_temp
})

# Save spike times to a separate file for reference
spike_times_df = pd.DataFrame({
    'spike_number': range(1, NUM_SPIKES + 1),
    'start_time': [timestamps[start] for start in spike_starts],
    'peak_time': [timestamps[start + SPIKE_DURATION//2] for start in spike_starts]
})

# Save to CSV files
df.to_csv('temperature_data.csv', index=False)
spike_times_df.to_csv('spike_times.csv', index=False)
