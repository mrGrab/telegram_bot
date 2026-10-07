import requests

from core.logger import logger


class SensuClient:
    """Handles communication with the Sensu Go API"""

    def __init__(self, url: str, api_key: str, namespace: str):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.namespace = namespace
        logger.debug(
            f"SensuClient initialized for namespace '{namespace}' at {self.url}"
        )

    def _send_request(
        self,
        url: str,
        data: dict | None = None,
        headers: dict | None = None,
        method: str = "GET",
        timeout: float = 10,
    ):
        """Internal method to send requests to the Sensu API"""
        if headers is None:
            headers = {}
        if data is None:
            data = {}

        headers.update(
            {"Authorization": f"Key {self.api_key}", "Content-Type": "application/json"}
        )

        try:
            r = requests.request(
                method=method, url=url, headers=headers, json=data, timeout=timeout
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            logger.error(f"Sensu API returned error {status}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Sensu API request failed: {e}")

    def execute_check(self, check_name: str) -> int | None:
        """Executes a Sensu check and returns the 'issued' timestamp"""
        url = f"{self.url}/api/core/v2/namespaces/{self.namespace}/checks/{check_name}/execute"
        data = {"check": check_name}
        response = self._send_request(url, data, method="POST")
        if response is None:
            return None
        if not isinstance(response, dict) or "issued" not in response:
            raise ValueError("Sensu execute response has no issued timestamp")
        return response["issued"]

    def get_event_check(
        self, check_name: str, entity: str, timeout: float = 10
    ) -> dict | None:
        """Retrieves a specific check event"""
        url = f"{self.url}/api/core/v2/namespaces/{self.namespace}/events/{entity}/{check_name}"
        return self._send_request(url, timeout=timeout)
