from typing import Literal

from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    workload_type: Literal[
        "web",
        "database",
        "api",
        "batch",
        "general"
    ]

    expected_users: int = Field(gt=0)

    environment: Literal[
        "development",
        "testing",
        "production"
    ]

    high_availability: bool = False


class RecommendationResponse(BaseModel):
    recommended_vcpu: int
    recommended_ram_gb: int
    recommended_disk_gb: int
    vm_count: int