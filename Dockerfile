FROM python:3.11.18-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/app/data \
    PSB_STATE_DIR=/app/state

WORKDIR /app
RUN groupadd --system --gid 10001 psb \
    && useradd --system --uid 10001 --gid psb --no-create-home psb \
    && install -d -m 0700 -o 10001 -g 10001 /app/state

# Runtime is Python's standard library; no downloaded packages or model weights.
COPY --chown=10001:10001 *.py requirements.txt ./
COPY --chown=10001:10001 static/ ./static/
COPY --chown=10001:10001 scripts/ ./scripts/
COPY --chown=10001:10001 data/knowledge-base.zip data/demo_questions.csv ./data/

USER 10001:10001
VOLUME ["/app/state"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2).read()"

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
