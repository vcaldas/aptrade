# This Dockerfile builds the API only.

FROM python:3.12
ARG FLASK_ENV
ENV FLASK_APP wsgi.py

COPY app app
COPY boot.sh wsgi.py ./
RUN chmod a+x boot.sh

RUN pip install -r /app/requirements.txt

EXPOSE 5000
ENTRYPOINT ["./boot.sh"]
