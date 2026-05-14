FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -e . && pip install --no-cache-dir gunicorn

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "price_forecast.app:create_app()"]
