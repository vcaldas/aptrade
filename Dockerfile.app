# This Dockerfile builds the API only.

FROM python:3.12
WORKDIR /app
ARG FLASK_ENV
COPY app ./
RUN pip install -r ./requirements.txt

EXPOSE 5000
CMD ["gunicorn", "-b", ":5000", "api:app"]
