FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CLAIMS_REVIEW_DEMO_DIR=/data/analysis_results

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY deploy/docker/review_requirements.py /build/review_requirements.py
RUN python /build/review_requirements.py /app/pyproject.toml > /build/requirements.txt \
    && python -m pip install --no-cache-dir -r /build/requirements.txt \
    && useradd --create-home --uid 10001 review

COPY src/ /app/src/
USER review
EXPOSE 8000
CMD ["uvicorn", "api.review_app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
