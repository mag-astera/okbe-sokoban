FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY runtime runtime
COPY levels levels
COPY static static
COPY app.py player_metrics.py .
USER 65534:65534
EXPOSE 8080
CMD ["python", "app.py"]
