FROM python:3.10 AS base

WORKDIR /app

RUN apt-get update && apt-get install -y git
RUN pip install poetry --no-cache-dir

COPY pyproject.toml poetry.lock /app/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN pip install -r requirements.txt --no-cache-dir

COPY . /app

CMD ["python", "main.py"]
