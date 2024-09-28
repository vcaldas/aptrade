#!/bin/bash

#  Paths recommended by pstrade. See https://github.com/robcarver17/pysystemtrade/blob/master/docs/production.md#quick-start-guide

echo -n "Checking if we can write to ~/.profile..."
if [ -e "${HOME}/.profile" ]; then
    if [ ! -w "${HOME}/.profile" ]; then
        echo -e "\xe2\x9a\xa0\xef\xb8\x8f"
        echo
        echo "    ** WARNING **"
        echo "    Your ~/.profile file seems to be non-writable. Will try to fix this with sudo"

        sudo chown ${USER}:staff ~/.profile
        echo -n "    Checking if fix worked..."
        if [ ! -w "${HOME}/.profile" ]; then
            echo -e "\xE2\x9D\x8C"
            echo 
            echo "    ** ERROR **"
            echo "    Something went wrong, we still can't write to ~/.profile"
            echo "    Please ensure this file is owned by your user and writeable by your user"
            exit 1
        fi
        echo -e "\xE2\x9C\x85"
    else
        echo -e "\xE2\x9C\x85"
    fi
else
    echo -e "\xE2\x9C\x85"
fi

# List of directories

directories=("$HOME/data/mongodb/"
        "$HOME/data/echos/"
        "$HOME/data/mongo_dump"
        "$HOME/data/backups_csv"
        "$HOME/data/backtests"
        "$HOME/data/reports"
        "$HOME/backup/mongo_backup"
        )


for dir in "${directories[@]}"
do
    if [[ ! -e $dir ]]; then
        echo "Creating $dir" 1>&2
        mkdir -p $dir
    elif [[ ! -d $dir ]]; then
        echo "$dir already exists but is not a directory" 1>&2
    fi
done

## Add variables to ./profile

# TODO: Check enviornment vars exist
# TODO: add env vars to .profile
# TODO: Create venvs for python 
#  check if variable exists in ./profile

## Verify if venv exist

if [ ! -d "$HOME/aptrade/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $HOME/aptrade/venv/
fi

## Activate Environment
source $HOME/aptrade/venv/bin/activate
## Install requirements
pip install -r $HOME/aptrade/requirements-dev.txt
## Install Pysystemtrade on development mode
python $HOME/aptrade/setup.py develop
python -m ipykernel install --user --name=aptrade

PYSYS_DATA_DIR=$HOME/data/
MONGO_DATA=$PYSYS_DATA_DIR/mongodb
PYSYS_CODE=$PWD/
SCRIPT_PATH=$PWD/sysproduction/linux/scripts
ECHO_PATH=$PYSYS_DATA_DIR/echos
MONGO_BACKUP_PATH=$HOME/backup/mongo_backup
APT_SCRIPTS=$HOME/aptrade/scripts
PYSYS_PRIVATE_CONFIG_DIR=$HOME/aptrade/private
PYSYS_LOGGING_CONFIG=$ECHO_PATH/syslogging.logging_prod.yaml


# Verify if variables exist in .profile
if ! grep -q "MONGO_DATA=" ~/.profile; then
    echo "export MONGO_DATA=${MONGO_DATA}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "PYSYS_CODE=" ~/.profile; then
    echo "export PYSYS_CODE=${PYSYS_CODE}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "SCRIPT_PATH=" ~/.profile; then
    echo "export SCRIPT_PATH=${SCRIPT_PATH}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "ECHO_PATH=" ~/.profile; then
    echo "export ECHO_PATH=${ECHO_PATH}   # ADDED BY APTRADE" >> ~/.profile
fi


if ! grep -q "APT_SCRIPTS=" ~/.profile; then
    echo "export APT_SCRIPTS=${APT_SCRIPTS}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "MONGO_BACKUP_PATH=" ~/.profile; then
    echo "export MONGO_BACKUP_PATH=${MONGO_BACKUP_PATH}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "PYSYS_PRIVATE_CONFIG_DIR=" ~/.profile; then
    echo "export PYSYS_PRIVATE_CONFIG_DIR=${PYSYS_PRIVATE_CONFIG_DIR}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "PYSYS_DATA_DIR=" ~/.profile; then
    echo "export PYSYS_DATA_DIR=${PYSYS_DATA_DIR}   # ADDED BY APTRADE" >> ~/.profile
fi

if ! grep -q "PYSYS_LOGGING_CONFIG=" ~/.profile; then
    echo "export PYSYS_LOGGING_CONFIG=${PYSYS_LOGGING_CONFIG}   # ADDED BY APTRADE" >> ~/.profile
fi

# Check if SCRIPT_PATH is in PATH
if ! echo "$PATH" | grep -q "$SCRIPT_PATH"; then
    echo "Adding SCRIPT_PATH to PATH..."
    echo "export PATH=\$PATH:$SCRIPT_PATH   # ADDED BY APTRADE" >> ~/.profile
fi

