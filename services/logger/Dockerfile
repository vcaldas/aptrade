FROM python:3.10-slim

WORKDIR /application

# RUN apt-get update

RUN pip install --upgrade pip && pip install pandas pyyaml
# COPY syslogging syslogging
# COPY syslogdiag syslogdiag
# COPY syscore syscore
# COPY sysdata sysdata

# RUN touch __init__.py
ENV PYTHONPATH=/syslogging:
ENV PYTHONPATH=/syslogdiag:
ENV PYTHONPATH=/syscore:
ENV PYTHONPATH=/sysdata:

COPY . . 

RUN ls -l
# EXPOSE 6020
CMD ["python", "syslogging/server.py", "--file",  "/application/output/pysystemtrade.log"] 