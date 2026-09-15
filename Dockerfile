FROM python:3.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-runtime.txt

COPY backend/app/ ./backend/app/
COPY ML/src/predict.py ML/src/build_dataset.py ML/src/preprocess.py ./ML/src/
COPY ML/models/reference_v1/cpu_reference.joblib ML/models/reference_v1/mem_reference.joblib ./ML/models/reference_v1/
COPY scripts/check_app.py ./scripts/check_app.py

USER 10001:10001
# Validate saved model compatibility on Linux during the build; no training.
RUN python -m ML.src.predict --instance-num 2 --plan-cpu 50 --plan-mem 0.005 > /dev/null
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python scripts/check_app.py --health-only
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
