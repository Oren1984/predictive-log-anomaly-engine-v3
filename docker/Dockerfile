# Dockerfile

# Purpose: Dockerfile for the predictive-log-anomaly-engine project, defining the image build process.

# Input: Python source code, scripts, and dependencies.

# Output: A Docker image that can run the predictive-log-anomaly-engine application.

# Used by: Docker and Docker Compose to build and run the application.


# # Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install curl for HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
# requirements.txt lives under requirements/ in the repo root.
COPY requirements/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source (build context must be repo root — see docker-compose.yml)
COPY src/ src/
COPY scripts/ scripts/
# templates/ holds the FastAPI HTML/Jinja2 UI templates served by the API.
COPY templates/ templates/

# Copy the template vocabulary needed by the v2 _V2LogTokenizer at runtime.
# data/intermediate/ is excluded from the Docker build context by .dockerignore,
# so templates.csv is injected via a read-only volume mount instead.
# This RUN just ensures the target directory exists so the mount succeeds.
RUN mkdir -p data/intermediate

# Create empty model/artifact dirs; they are populated via volume mounts
# in docker-compose.yml (../models and ../artifacts).  This means the image
# builds successfully in CI even when those gitignored directories are absent.
RUN mkdir -p models artifacts

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["python", "-m", "uvicorn", "src.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]