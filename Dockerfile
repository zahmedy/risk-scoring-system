FROM python:3.11-slim

WORKDIR /app

# (Optional but common) system build tools for scientific libs
RUN apt-get update && apt-get install -y build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project metadata first (better caching)
COPY pyproject.toml ./

# Copy package + artifacts/configs 
COPY src/ src/
COPY artifacts/ artifacts/
COPY configs/ configs/
COPY README.md README.md

# Install the package
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "risk_system.api:app", "--host", "0.0.0.0", "--port", "8080"]
