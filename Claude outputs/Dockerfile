FROM python:3.11-slim

# Non-root user (Hugging Face Spaces runs containers as uid 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install packages first so Docker can cache this slow step (CPU-only torch)
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the whole project (index.html, app.js, style.css, backend/, analytics/ ...)
COPY --chown=user:user . /app
RUN chown -R user:user /app
USER user

# Hugging Face Spaces expects port 7860; Render sets its own PORT at runtime
ENV HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

CMD ["python", "backend/server.py"]
