# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.15
FROM python:${PYTHON_VERSION}-slim as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p /app/staticfiles && chmod 755 /app/staticfiles

RUN python manage.py collectstatic --noinput

EXPOSE 8000

CMD gunicorn django_project.wsgi:application --bind 0.0.0.0:8000