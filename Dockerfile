FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent

ENV APP_NAME=agent-001 PORT=9000
EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:9000/health || exit 1

# non-root
RUN useradd -m app && chown -R app:app /app
USER app

CMD ["sh", "-c", "uvicorn agent.main:app --host 0.0.0.0 --port ${PORT}"]