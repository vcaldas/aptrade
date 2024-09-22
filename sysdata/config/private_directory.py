import os

from sysdata import BASE_PATH
DEFAULT_PRIVATE_DIR = "private"
PRIVATE_CONFIG_DIR_ENV_VAR = "PYSYS_PRIVATE_CONFIG_DIR"


def get_full_path_for_private_config(filename: str):
    if os.getenv(PRIVATE_CONFIG_DIR_ENV_VAR):
        _directory  = os.environ[PRIVATE_CONFIG_DIR_ENV_VAR]
    else:
        _directory  = os.path.join(BASE_PATH, DEFAULT_PRIVATE_DIR)
    private_config_path = os.path.join(_directory, filename)
    return private_config_path
