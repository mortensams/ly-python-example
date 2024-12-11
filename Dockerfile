FROM python:3.9-slim-bullseye

# Install system dependencies including X11 and browser dependencies
RUN apt-get update && apt-get install -y \
    curl \
    xvfb \
    firefox-esr \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -r appuser

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY api.py .
COPY temperature_viewer.py .
COPY start_services.sh .
COPY temperature_data.csv .

# Make startup script executable
RUN chmod +x start_services.sh

# Switch to non-root user
USER appuser

# Set up virtual display
ENV DISPLAY=:99

# Expose ports
EXPOSE 8000

# Start services
CMD Xvfb :99 -screen 0 1024x768x16 & \
    ./start_services.sh