import importlib
import pkgutil

from fastapi import APIRouter


def __getattr_default(_object, _attribute, default):
    if hasattr(_object, _attribute):
        return getattr(_object, _attribute)
    else:
        return default


def get_dynamic_api_details() -> tuple[APIRouter, list[dict]]:
    """Construct the APIRouter for our api endpoint

    Returns:
        APIRouter that has a version layer (/v1) and under that URL has all of the API
        endpoints you include
    """
    versions: dict[str, list[APIRouter]] = {}
    tags_descriptions = []

    # Recursively start going through all submodules / packages and find where we
    # have API_ENDPOINT_ROUTER defined at package level
    pkg_list = [importlib.import_module(".", ".".join(__name__.split(".")[:-1]))]
    while len(pkg_list):
        current_pkg = pkg_list.pop(0)

        for _, modname, ispkg in pkgutil.iter_modules(
            current_pkg.__path__, current_pkg.__name__ + "."
        ):
            if ispkg:
                # Check if there exists the API_ENDPOINT_ROUTER
                # If that is the case, add the details, otherwise, recursively go
                # through this package again

                api_endpoint_module = importlib.import_module(modname)

                if "API_ENDPOINT_ROUTER" in dir(api_endpoint_module):
                    api_router = getattr(api_endpoint_module, "API_ENDPOINT_ROUTER")

                    api_version = __getattr_default(
                        api_endpoint_module, "API_ENDPOINT_VERSION", ""
                    )
                    api_disabled = __getattr_default(
                        api_endpoint_module, "API_ENDPOINT_DISABLED", False
                    )
                    api_tags_description = __getattr_default(
                        api_endpoint_module, "API_ENDPOINT_TAGS_DESCRIPTION", []
                    )

                    if not api_disabled:
                        if api_version not in versions:
                            versions[api_version] = []

                        versions[api_version].append(api_router)
                        tags_descriptions.extend(api_tags_description)

                else:
                    pkg_list.append(api_endpoint_module)

    api_router = APIRouter()

    for _version, _api_routers in versions.items():
        router_to_add_routers_to = api_router
        if _version != "":
            router_to_add_routers_to = APIRouter()
            if not _version.startswith("/"):
                _version = f"/{_version}"

        for _api_router in _api_routers:
            router_to_add_routers_to.include_router(_api_router)

        if _version != "":
            api_router.include_router(router_to_add_routers_to, prefix=_version)

    return api_router, tags_descriptions
