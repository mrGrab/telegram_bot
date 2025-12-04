import requests
from typing import Dict, Optional
from core.logger import logger


class SensuClient(object):
    """Handles communication with the Sensu Go API"""

    def __init__(self, url: str, api_key: str, namespace: str):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.namespace = namespace
        logger.debug(
            f"SensuClient initialized for namespace '{namespace}' at {self.url}"
        )

    def _send_request(self,
                      url: str,
                      data: Optional[Dict] = None,
                      headers: Optional[Dict] = None,
                      method: str = "GET"):
        """Internal method to send requests to the Sensu API"""
        if headers is None:
            headers = {}
        if data is None:
            data = {}

        headers.update({
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        })

        try:
            r = requests.request(method=method,
                                 url=url,
                                 headers=headers,
                                 json=data,
                                 timeout=10)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Sensu API returned error {r.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Sensu API request failed: {e}")

    def execute_check(self, check_name: str) -> Optional[int]:
        """Executes a Sensu check and returns the 'issued' timestamp"""
        url = f"{self.url}/api/core/v2/namespaces/{self.namespace}/checks/{check_name}/execute"
        data = {"check": check_name}
        response = self._send_request(url, data, method="POST")
        return response.get("issued") if response else None

    def get_event_check(self, check_name: str, entity: str) -> Optional[Dict]:
        """Retrieves a specific check event"""
        url = f"{self.url}/api/core/v2/namespaces/{self.namespace}/events/{entity}/{check_name}"
        return self._send_request(url)
