FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

# Copy dependency files first (for better Docker layer caching)
COPY pyproject.toml /app/pyproject.toml

# Copy the entire src directory to maintain proper package structure
COPY src/ /app/src/

# Copy the main script
COPY token_benchmark_ray.py /app/token_benchmark_ray.py

# Install the package in editable mode
RUN pip install -e .

# Create a non-root user for security (Ubuntu syntax)
RUN useradd -m -s /bin/bash llmperf
USER llmperf

# Set the entrypoint to the token_benchmark_ray.py script
ENTRYPOINT ["python", "/app/token_benchmark_ray.py"]

# Default command (can be overridden)
CMD ["--help"]