FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Run as a non-root user (created first so we can chown to it).
RUN useradd --create-home --uid 10001 codas

# Install only the importable packages and their deps. examples/ and tests/ are not shipped.
COPY --chown=codas:codas pyproject.toml README.md ./
COPY --chown=codas:codas codas_core ./codas_core
COPY --chown=codas:codas codas_agents ./codas_agents
COPY --chown=codas:codas codas_service ./codas_service
RUN pip install --no-cache-dir ".[service,agent]"

# Create the uploads directory and set ownership so the non-root user can write to it.
RUN mkdir -p /app/.codas_runs/agent_uploads && chown -R codas:codas /app/.codas_runs

USER codas

# Cloud Run injects $PORT (defaults to 8080). Shell form so it expands.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn codas_service.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
