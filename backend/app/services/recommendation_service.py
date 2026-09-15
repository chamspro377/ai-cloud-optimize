from backend.app.schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
)


def recommend_resources(
    request: RecommendationRequest,
) -> RecommendationResponse:

    # Temporary recommendation.
    # Later this will be replaced by the ML model.

    vcpu = 2
    ram = 4
    disk = 50
    vm_count = 1

    if request.expected_users > 500:
        vcpu = 4
        ram = 8
        disk = 100

    if request.expected_users > 2000:
        vcpu = 8
        ram = 16
        disk = 200

    if request.environment == "production":
        disk += 50

    if request.high_availability:
        vm_count = 2

    return RecommendationResponse(
        recommended_vcpu=vcpu,
        recommended_ram_gb=ram,
        recommended_disk_gb=disk,
        vm_count=vm_count,
    )