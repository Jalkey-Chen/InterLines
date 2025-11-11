FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_SYSTEM_PYTHON=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential git && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN useradd -ms /bin/bash appuser

FROM base AS deps
WORKDIR /app
COPY pyproject.toml ./
COPY uv.lock ./uv.lock || true
RUN uv sync --all-extras --no-dev

FROM base AS runtime
WORKDIR /app
COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --from=deps /root/.cache/uv /root/.cache/uv
RUN uv sync --all-extras
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser docs ./docs
COPY --chown=appuser:appuser schemas ./schemas
COPY --chown=appuser:appuser .env.example ./
RUN mkdir -p artifacts/cards artifacts/reports artifacts/trace logs && chown -R appuser:appuser /app
USER appuser
ENV INTERLINES_ENV=production LOG_LEVEL=INFO PYTHONPATH=/app/src
EXPOSE 8000 8501
CMD ["uv","run","interlines-api"]
