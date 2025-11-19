FROM python:3.13-slim-bookworm


# Set correct working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# # Set Python path so imports work correctly
# ENV PYTHONPATH=/app

# # Run app with uvicorn
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# # Expose port
# EXPOSE 8000