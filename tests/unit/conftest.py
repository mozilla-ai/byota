import json
import pytest

from pathlib import Path


def load_json(path: Path) -> dict:
    with Path.open(path) as file:
        return json.load(file)


def resources_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def paginated_data() -> dict:
    return load_json(resources_dir() / "mastodon_data.json")
