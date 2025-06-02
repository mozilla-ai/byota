import json
import os
from pathlib import Path
from pydantic import BaseModel, field_validator


class MastodonAccount(BaseModel):
    name: str
    account_type: str = "mastodon"
    MASTODON_ACCESS_TOKEN: str
    MASTODON_API_BASE_URL: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if " " in v:
            raise ValueError("Account name cannot contain spaces")
        return v


class LegacyConfig(BaseModel):
    MASTODON_ACCESS_TOKEN: str
    MASTODON_API_BASE_URL: str


def load_accounts(cred_filename: str = "auth.json") -> list[MastodonAccount]:
    """
    Load account configurations from file or environment variables.

    Args:
        cred_filename: Path to the JSON credentials file

    Returns:
        List of MastodonAccount objects

    Raises:
        FileNotFoundError: If credentials file is missing and env vars are not set
        ValueError: If configuration format is invalid
    """
    if Path(cred_filename).is_file():
        with Path(cred_filename).open() as f:
            raw_config = json.load(f)

        if isinstance(raw_config, list):
            # New multi-account format
            accounts = [MastodonAccount(**account) for account in raw_config]
        else:
            # Legacy single account format
            legacy = LegacyConfig(**raw_config)
            accounts = [
                MastodonAccount(
                    name="Default",
                    MASTODON_ACCESS_TOKEN=legacy.MASTODON_ACCESS_TOKEN,
                    MASTODON_API_BASE_URL=legacy.MASTODON_API_BASE_URL,
                )
            ]
    else:
        # Environment variables (single account)
        access_token = os.environ.get("MASTODON_ACCESS_TOKEN")
        api_base_url = os.environ.get("MASTODON_API_BASE_URL")

        if not access_token or not api_base_url:
            raise FileNotFoundError(
                f"Credentials file '{cred_filename}' not found and "
                "MASTODON_ACCESS_TOKEN/MASTODON_API_BASE_URL environment variables not set"
            )

        accounts = [
            MastodonAccount(
                name="Default",
                MASTODON_ACCESS_TOKEN=access_token,
                MASTODON_API_BASE_URL=api_base_url,
            )
        ]

    return accounts


def mock_account_list(name="Default") -> list[MastodonAccount]:
    return [
        MastodonAccount(
            name=name,
            MASTODON_ACCESS_TOKEN="",
            MASTODON_API_BASE_URL="",
        )
    ]
