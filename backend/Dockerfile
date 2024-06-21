FROM python:3.11-slim

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

WORKDIR /application/

RUN pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org" 
RUN pip config set global.timeout 900


RUN groupadd -g 1998 appuser && useradd --create-home -u 1998 -g appuser appuser \
  && chown -R appuser:appuser /application \
  && chown -R appuser:appuser /application



COPY --chown=appuser:appuser backend /application/backend
COPY --chown=appuser:appuser packages /application/packages


RUN pip install -r /application/backend/requirements.txt

RUN chmod a+x /application/backend/boot.sh

EXPOSE 8000
WORKDIR /application/backend

ENTRYPOINT ["bash", "/application/backend/boot.sh"]

CMD []