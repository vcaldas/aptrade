FROM python:3.11-slim

WORKDIR /webapp

COPY . .

RUN pip install -r requirements.txt
RUN pip install gunicorn pymysql cryptography

RUN chmod a+x boot.sh

ENV FLASK_APP server.py
RUN flask translate compile

EXPOSE 5000
ENTRYPOINT ["bash", "./boot.sh"]

CMD []