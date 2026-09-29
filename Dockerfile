FROM python:3.11-slim

WORKDIR /app

# Ensure Python doesn't buffer stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV TTA_ROTATIONS=4

# Install requirements first for caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (including backend/, models, frontend assets)
COPY . /app/

EXPOSE 8000

CMD ["python", "backend/server.py"]
