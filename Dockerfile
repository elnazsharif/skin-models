# ---------------------------------------------------------
# Glow AI Recommender – Optimized Dockerfile for Railway
# ---------------------------------------------------------

# 1️⃣ Use lightweight Python base image
FROM python:3.11-slim

# 2️⃣ Install minimal OS packages needed for YOLO and OpenCV
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 3️⃣ Set working directory inside the container
WORKDIR /app

# 4️⃣ Copy only requirements first (for Docker layer caching)
COPY requirements.txt .

# 5️⃣ Upgrade pip and install Python dependencies without cache
#    This speeds up installs and reduces image size
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy the rest of your project files (app.py, weights/, etc.)
COPY . .

# 7️⃣ Expose FastAPI port
EXPOSE 8000

# 8️⃣ Launch FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
