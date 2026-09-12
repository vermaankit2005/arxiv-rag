from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

APPLICATION_FILE = Path(__file__).parents[2] / "application.yaml"


def application_config() -> dict[str, Any]:
    load_dotenv()
    with APPLICATION_FILE.open(encoding="utf-8") as application_file:
        return yaml.safe_load(application_file)
