# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY run_pipeline.py run_pipeline.sh ./

RUN python -m pip install --upgrade pip \
    && python -m pip install .

RUN mkdir -p /app/data/raw /app/data/processed \
    /app/reports/metrics /app/reports/models /app/reports/candidates

ENTRYPOINT ["gprml-pipeline"]
CMD ["--help"]
