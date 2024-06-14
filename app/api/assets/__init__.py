"""API Router Definition for cats that allows for dynamic including in fastapi

The following items need to be defined at python package level for our loader to be
able to dynamically load it:

API_ENDPOINT_ROUTER  (MANDATORY)
    (str) the FastAPI APIRouter object for the endpoint

API_ENDPOINT_VERSION  (OPTIONAL)
   (str) The version prefix for the API endpoint (e.g. v1 or v2)
         Defaults to empty string

API_ENDPOINT_DISABLED  (OPTIONAL)
    (bool) Should this router be disabled.
           Defaults to False, meaning the router will be dynamically included

API_ENDPOINT_TAGS_DESCRIPTION  (OPTIONAL)
    (list[dict]) Dictionary holding the tags metadata. See the fastapi for more details
                 on the structure of this.
                 Basic example = [{"name": "tag_name", "description": "tag description"}]

                 Defaults to empty list
"""
from .assets_api_router import assets_router

API_ENDPOINT_ROUTER = assets_router
API_ENDPOINT_VERSION = "v1"
API_ENDPOINT_DISABLED = False
API_ENDPOINT_TAGS_DESCRIPTION = [{"name": "cats", "description": "Operations on cats"}]
