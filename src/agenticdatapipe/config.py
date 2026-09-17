"""Environment-backed settings shared by CLI, graph, and tools."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIKE_", env_file=".env", extra="ignore")

    feed_url: str = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
    system_id: str = "citibike-nyc"
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "bike.station_status.v1"
    kafka_group_id: str = "bike-pipeline-v1"
    data_dir: Path = Path("data")
    batch_size: int = Field(default=1000, ge=1)
    batch_timeout_seconds: float = Field(default=30, gt=0)
    poll_seconds: float = Field(default=60, ge=60)
    metadata_refresh_seconds: float = Field(default=3600, ge=60)
    stale_seconds: int = Field(default=900, ge=1)
    repair_limit: int = Field(default=2, ge=0, le=2)
    http_timeout_seconds: float = Field(default=20, gt=0)
    openai_model: str = "gpt-4.1-mini"
    fixture_mode: bool = False
    storage_backend: Literal["local", "delta"] = "local"
    s3_endpoint: str = "http://localhost:9000"
    s3_bucket: str = "bike-lake"
    s3_region: str = "us-east-1"
    s3_access_key_id: str = "minioadmin"
    s3_secret_access_key: str = "minioadmin"
    s3_allow_http: bool = True
    observations_table: str = "delta/station_observations"
    features_table: str = "delta/station_features"
    raw_prefix: str = "raw/gbfs"
    audit_prefix: str = "audit/runs"

    def s3_uri(self, key: str) -> str:
        return f"s3://{self.s3_bucket}/{key.lstrip('/')}"

    def delta_storage_options(self) -> dict[str, str]:
        return {
            "AWS_ACCESS_KEY_ID": self.s3_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.s3_secret_access_key,
            "AWS_REGION": self.s3_region,
            "AWS_ENDPOINT_URL": self.s3_endpoint,
            "AWS_VIRTUAL_HOSTED_STYLE_REQUEST": "false",
            "allow_http": str(self.s3_allow_http).lower(),
            "conditional_put": "etag",
        }
