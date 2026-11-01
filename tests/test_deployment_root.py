import unittest

from fastapi.testclient import TestClient

from src.api.config import ApiConfig
from src.api.main import create_app


class PublicRootTests(unittest.TestCase):
    def test_public_root_lists_documented_api_links_without_internal_details(self):
        client = TestClient(create_app(ApiConfig()))
        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["name"], "LinearGridTransformer")
        self.assertEqual(body["links"], {
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "api": "/api/v1",
        })
        serialized = response.text.lower()
        self.assertNotIn("path", body)
        self.assertNotIn("/users/", serialized)
        self.assertNotIn("password", serialized)
        self.assertNotIn("secret", serialized)


if __name__ == "__main__":
    unittest.main()
