# Use an official Python image from Docker Hub
FROM python:3.11-slim

# Install some system packages that YOLO and OpenCV need
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set a working folder inside the container
WORKDIR /app

# Copy your requirements file first (so it can install dependencies)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your other files (app.py, weights, etc.)
COPY . .

# Tell Railway which port your FastAPI will run on
EXPOSE 8000

# Start your app when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

