import time
import random
import requests
from typing import Any, Dict, Optional

# Simple exponential back‑off helper for GET requests.
# Returns the ``Response`` object or raises ``requests.RequestException`` after retries.

def http_get_with_retry(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10,
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> requests.Response:
    """Perform a GET request with exponential back‑off.

    Parameters
    ----------
    url: str
        Target URL.
    headers: dict | None
        Optional HTTP headers.
    timeout: int
        Socket timeout in seconds.
    retries: int
        Number of attempts (first attempt + ``retries``‑1 retries).
    backoff_factor: float
        Base delay in seconds; the actual wait is ``backoff_factor * (2 ** attempt)``.
    """
    attempt = 0
    while True:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.RequestException, requests.HTTPError) as exc:
            attempt += 1
            if attempt > retries:
                raise  # propagate the last exception
            # jitter to avoid thundering herd
            sleep_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
            time.sleep(sleep_time)
            continue
