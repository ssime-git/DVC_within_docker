# Use a Python base image
FROM python:3.9-slim

# Install git (needed by DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install DVC and Python dependencies for our stages
# Add remote deps if needed, e.g., dvc[s3]
RUN pip install --no-cache-dir \
    dvc \
    pandas \
    scikit-learn

# Set the working directory inside the container
WORKDIR /workspace