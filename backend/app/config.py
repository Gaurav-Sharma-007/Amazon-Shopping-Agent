from __future__ import annotations

import configparser
import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    app_name: str = "Amazon Preference Recommender"
    aws_region: str = "ap-south-1"
    dynamodb_table: str = "amazon-product-recommendations"
    use_dynamodb: bool = True
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    amazon_domain: str = "https://www.amazon.in"
    scraper_max_pages: int = Field(default=1, ge=1, le=5)
    scraper_max_results: int = Field(default=12, ge=1, le=48)
    scraper_timeout_ms: int = Field(default=25_000, ge=5_000, le=90_000)
    local_data_dir: Path = ROOT_DIR / "data"


def _coerce_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> Settings:
    parser = configparser.ConfigParser()
    config_path = ROOT_DIR / "config.ini"
    if config_path.exists():
        parser.read(config_path)

    app = parser["app"] if parser.has_section("app") else {}
    aws = parser["aws"] if parser.has_section("aws") else {}
    scraper = parser["scraper"] if parser.has_section("scraper") else {}

    return Settings(
        app_name=os.getenv("APP_NAME", app.get("name", Settings().app_name)),
        aws_region=os.getenv("AWS_REGION", aws.get("region", Settings().aws_region)),
        dynamodb_table=os.getenv(
            "DYNAMODB_TABLE", aws.get("dynamodb_table", Settings().dynamodb_table)
        ),
        use_dynamodb=_coerce_bool(
            os.getenv("USE_DYNAMODB", aws.get("use_dynamodb")), Settings().use_dynamodb
        ),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID", aws.get("bedrock_model_id", Settings().bedrock_model_id)
        ),
        amazon_domain=os.getenv(
            "AMAZON_DOMAIN", scraper.get("amazon_domain", Settings().amazon_domain)
        ).rstrip("/"),
        scraper_max_pages=int(
            os.getenv("SCRAPER_MAX_PAGES", scraper.get("max_pages", Settings().scraper_max_pages))
        ),
        scraper_max_results=int(
            os.getenv(
                "SCRAPER_MAX_RESULTS",
                scraper.get("max_results", Settings().scraper_max_results),
            )
        ),
        scraper_timeout_ms=int(
            os.getenv(
                "SCRAPER_TIMEOUT_MS",
                scraper.get("timeout_ms", Settings().scraper_timeout_ms),
            )
        ),
    )

