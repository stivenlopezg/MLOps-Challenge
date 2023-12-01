FROM python:3.9-slim
LABEL authors="stlopez"

WORKDIR /app
COPY pyproject.toml poetry.lock /app/

RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry install --no-interaction --no-ansi

COPY app /app

ENTRYPOINT ["python3", "app/steps/run_step.py"]