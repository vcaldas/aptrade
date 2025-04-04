#!/bin/bash
# this script is used to boot a Docker container

if [ "$FLASK_ENV" = "development" ]; then
    echo "FLASK_ENV is set to development"
    exec flask --app wsgi run --host=0.0.0.0

else
    echo "FLASK_ENV is not set to development"
    exec gunicorn -b :5000 --access-logfile - --error-logfile - wsgi:app

fi

# while true; do
#     flask db upgrade
#     if [[ "$?" == "0" ]]; then
#         break
#     fi
#     echo Deploy command failed, retrying in 5 secs...
#     sleep 5
# done
