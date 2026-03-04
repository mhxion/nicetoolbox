from typing import Dict

from pydantic import BaseModel, RootModel


class AssetMetadata(BaseModel):
    """Metadata for a single asset."""

    url: str


class AssetManifest(RootModel[Dict[str, AssetMetadata]]):
    """
    Validation schema for asset_manifest.toml.
    Validates a dictionary where the key is the relative path (str)
    and the value contains the asset metadata (url).
    """

    pass
