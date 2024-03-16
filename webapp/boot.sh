#!/bin/bash
# this script is used to boot a Docker container

if [ "$DATABASE" = "postgres" ]
    then
        echo "Waiting for postgres..."

        while ! nc -z $DJANGO_DB_HOST $DJANGO_DB_PORT; do
        sleep 0.1
        done

        echo "PostgreSQL started"
fi

python manage.py migrate
python manage.py makesuperuser
python manage.py runserver  0.0.0.0:8000