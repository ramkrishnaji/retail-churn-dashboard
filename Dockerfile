# Use official Python slim image for a lightweight footprint
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies (needed for packages like building wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code into the container
COPY . .

# Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Make start.sh executable
RUN chmod +x start.sh

# Run the startup script
CMD ["./start.sh"]
