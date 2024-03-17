FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install -r requirements.txt

RUN chmod a+x boot.sh

EXPOSE 8000
ENTRYPOINT ["bash", "./boot.sh"]

CMD []