import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for 48 hours with 1-second intervals
start_time = datetime(2024, 1, 1, 0, 0, 0)
timestamps = [start_time + timedelta(seconds=x) for x in range(172800)]  # 48 hours * 3600 seconds

# Generate ambient temperature with daily cycles
time_hours = np.arange(172800) / 3600  # Convert seconds to hours
base_temp = 22  # Base indoor temperature
daily_variation = 1.5  # Daily temperature variation amplitude
noise_amplitude = 0.2  # Random noise amplitude

# Create smooth daily temperature cycle with some noise
ambient_temp = (base_temp + 
                daily_variation * np.sin(2 * np.pi * time_hours / 24) +  # Daily cycle
                noise_amplitude * np.random.randn(172800))  # Random noise

# Generate device temperature that correlates with ambient temperature
base_device_temp = 55  # Base device temperature
device_temp_factor = 2  # How much ambient temperature changes affect device
device_noise = 0.5  # Additional noise for device temperature

device_temp = (base_device_temp + 
              device_temp_factor * (ambient_temp - base_temp) +  # Correlation with ambient
              device_noise * np.random.randn(172800))  # Random noise

# Add 8 random overheating spikes
num_spikes = 8
spike_duration = 300  # 5 minutes in seconds
min_gap = 3600  # Minimum 1 hour between spikes

# Generate random spike start times (ensuring they don't overlap)
spike_starts = []
while len(spike_starts) < num_spikes:
    new_start = np.random.randint(0, 172800 - spike_duration)
    # Check if new spike is far enough from existing spikes
    if not any(abs(new_start - existing) < min_gap for existing in spike_starts):
        spike_starts.append(new_start)

# Sort spike starts for better readability in data
spike_starts.sort()

# Add each spike to the device temperature
for start in spike_starts:
    # Generate a random peak temperature between 70 and 80 degrees
    peak_temp = np.random.uniform(70, 80)
    
    # Create a smooth spike using a half sine wave
    for i in range(spike_duration):
        position = i / spike_duration
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
    'spike_number': range(1, num_spikes + 1),
    'start_time': [timestamps[start] for start in spike_starts],
    'peak_time': [timestamps[start + spike_duration//2] for start in spike_starts]
})
spike_times_df.to_csv('/users/mortensams/claude/python-aggregate/spike_times.csv', index=False)

# Save to CSV
df.to_csv('/users/mortensams/claude/python-aggregate/temperature_data.csv', index=False)
