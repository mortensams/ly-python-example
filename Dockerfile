# Use Python slim-bullseye as base image for security
FROM python:3.9-slim-bullseye

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
COPY temperature_data.csv .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "api.py"]
