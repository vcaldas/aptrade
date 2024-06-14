from typing import Optional

from pydantic import BaseModel


class Asset(BaseModel):
    asset_id: Optional[int] = None
    symbol: str
    name: str
