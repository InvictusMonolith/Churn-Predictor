# 1. Start from a lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first (enables Docker layer caching)
COPY requirements.txt .

# 4. Install dependencies inside container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app code and .joblib files
COPY . .

# 6. Expose the port FastAPI will run on
EXPOSE 8000

# 7. Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]