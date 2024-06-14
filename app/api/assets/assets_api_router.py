from fastapi import APIRouter

from .models import Asset
from .data import DATA_STOCK

assets_router = APIRouter(
    prefix="/asset",
    tags=["asset"],
)


@assets_router.get("/", response_model=list[Asset])
def read_items() -> list[Asset]:
    """Retrieve a list of all of the cats currently available

    Per cat the following information will be returned:

    - **asset_id**: The internal id used to store this asset
    - **symbol**: The symbol of the asset
    - **name**: Name of the asset

    Returns:
        List of Stocks objects
    """
    return list(DATA_STOCK.values())
