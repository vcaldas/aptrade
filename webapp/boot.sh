#!/bin/bash
# this script is used to boot a Docker container
while true; do
    python manage.py migrate
    if [[ "$?" == "0" ]]; then
        break
    fi
    echo Deploy command failed, retrying in 5 secs...
    sleep 5
done
python manage.py runserver  0.0.0.0:8000