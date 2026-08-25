"""Thin wrapper around the GitHub REST API: auth, pagination, and rate-limit backoff."""

import time

import requests

API_ROOT = "https://api.github.com"


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    return session


def _sleep_for_rate_limit(response: requests.Response) -> bool:
    """If response indicates a rate limit was hit, sleep and return True. Otherwise return False."""
    if response.status_code not in (403, 429):
        return False

    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        # Secondary rate limit (abuse detection) - header gives seconds to wait.
        time.sleep(int(retry_after) + 1)
        return True

    remaining = response.headers.get("X-RateLimit-Remaining")
    reset_at = response.headers.get("X-RateLimit-Reset")
    if remaining == "0" and reset_at is not None:
        # Primary rate limit exhausted - sleep until the reset timestamp.
        wait_seconds = max(int(reset_at) - int(time.time()), 0) + 1
        time.sleep(wait_seconds)
        return True

    return False


def paginated_get(session: requests.Session, url: str, params: dict | None = None):
    """Yield items one at a time from a paginated GitHub REST list endpoint.

    Follows the `Link: rel="next"` header until exhausted. Retries in place on
    rate-limit responses (403/429) rather than raising, since these are expected
    for a bulk pull of this size.
    """
    next_url = url
    next_params = params

    while next_url is not None:
        response = session.get(next_url, params=next_params)

        if _sleep_for_rate_limit(response):
            continue  # retry the same URL/params after sleeping

        response.raise_for_status()
        yield from response.json()

        next_url = response.links.get("next", {}).get("url")
        next_params = None  # subsequent URLs from Link header already include query params
