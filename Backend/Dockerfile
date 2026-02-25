FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for XGBoost and SQLite
RUN apt-get update && apt-get install -y \
    libgomp1 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
