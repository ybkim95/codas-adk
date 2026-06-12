FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the importable packages and their deps. examples/ and tests/ are not shipped.
COPY pyproject.toml README.md ./
COPY codas_core ./codas_core
COPY codas_agents ./codas_agents
COPY codas_service ./codas_service
RUN pip install --no-cache-dir ".[service,agent]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 codas
USER codas

# Cloud Run injects $PORT (defaults to 8080). Shell form so it expands.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn codas_service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
