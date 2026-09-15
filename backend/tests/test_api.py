import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.app.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.request = dict(workload_type="general", expected_users=100,
                            environment="development", high_availability=False)

    def test_page_and_health(self):
        self.assertIn("AI Cloud Optimizer", self.client.get("/").text)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_rule_boundaries_and_terraform_export(self):
        for users, cpu in ((500, 2), (501, 4), (2000, 4), (2001, 8)):
            self.request["expected_users"] = users
            self.assertEqual(self.client.post("/recommend", json=self.request).json()["recommended_vcpu"], cpu)
        self.request.update(environment="production", high_availability=True)
        response = self.client.post("/terraform/variables", json=self.request)
        self.assertEqual(response.json(), dict(vcpu=8, memory_mb=16384, disk_gb=250, vm_count=2))
        self.assertIn("attachment", response.headers["content-disposition"])

    def test_invalid_inputs(self):
        self.request["expected_users"] = 0
        self.assertEqual(self.client.post("/recommend", json=self.request).status_code, 422)
        self.assertEqual(self.client.post("/predict", json=dict(instance_num=1.5, plan_cpu=50, plan_mem=.005)).status_code, 422)

    def test_missing_models_is_service_unavailable(self):
        with patch("backend.app.main.predict_resources", side_effect=FileNotFoundError):
            self.assertEqual(self.client.post("/predict", json=dict(instance_num=2, plan_cpu=50, plan_mem=.005)).status_code, 503)

    def test_real_saved_models(self):
        response = self.client.post("/predict", json=dict(instance_num=2, plan_cpu=50, plan_mem=.005))
        self.assertEqual(response.status_code, 200, response.text)
        for resource in ("cpu", "mem"):
            values = response.json()["predictions"][resource]["values"]
            self.assertGreaterEqual(values[f"{resource}_observed_peak"], values[f"{resource}_mean_of_attempt_averages"])


if __name__ == "__main__":
    unittest.main()
