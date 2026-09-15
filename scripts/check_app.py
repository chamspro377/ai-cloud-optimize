"""Check the running demo using Python's standard library; no training."""
import argparse
import json
from urllib.request import Request, urlopen


def check(base_url, health_only=False):
    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(base_url.rstrip("/") + path, data=data,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=4 if health_only else 60) as response:
            return json.load(response)

    health = request("/health")
    if health.get("status") != "healthy" or not health.get("models_available"):
        raise RuntimeError("API or model files unavailable")
    if health_only:
        return
    payload = dict(workload_type="general", expected_users=100,
                   environment="development", high_availability=False)
    recommendation = request("/recommend", payload)
    if recommendation["recommended_vcpu"] != 2:
        raise RuntimeError("Unexpected recommendation")
    prediction = request("/predict", dict(instance_num=2, plan_cpu=50, plan_mem=0.005))
    for resource in ("cpu", "mem"):
        values = prediction["predictions"][resource]["values"]
        if not 0 <= values[resource + "_mean_of_attempt_averages"] <= values[resource + "_observed_peak"]:
            raise RuntimeError("Invalid prediction")
    if request("/terraform/variables", payload) != dict(vcpu=2, memory_mb=4096, disk_gb=50, vm_count=1):
        raise RuntimeError("Unexpected VM export")
    print("OK: health, recommendation, saved ML models and VM export")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--health-only", action="store_true")
    args = parser.parse_args()
    check(args.url, args.health_only)
