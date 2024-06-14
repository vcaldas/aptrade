from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """The settings for our fastapi application
    By using the BaseSettings from pydantic, each of these keys will be automatically
    updated if an environment variable with the same name exists
    """

    app_name: str = "APTrade"
    app_version: str = "LOCAL-UNKNOWN-VERSION"
    author: str = "Victor Caldas"
    custom: dict[str, str] = {}


def get_settings() -> Settings:
    """Convenience function to return a Settings object.
    This function will automatically update the captain_app_version setting to include
    prefix git- when running on dev platform. This will give a clear indication that
    the supplied version corresponds to a git-commit hash
    Returns:
        Settings object
    """
    settings = Settings()

    return settings
