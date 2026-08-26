from unittest.mock import Mock, patch

from src.ingest.github_client import _sleep_for_rate_limit, make_session, paginated_get


def test_make_session_sets_expected_headers():
    session = make_session("secret-token")

    assert session.headers["Authorization"] == "Bearer secret-token"
    assert session.headers["Accept"] == "application/vnd.github+json"
    assert session.headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_sleep_for_rate_limit_ignores_non_rate_limit_status():
    response = Mock(status_code=200, headers={})

    with patch("src.ingest.github_client.time.sleep") as mock_sleep:
        assert _sleep_for_rate_limit(response) is False
        mock_sleep.assert_not_called()


def test_sleep_for_rate_limit_uses_retry_after_header():
    response = Mock(status_code=403, headers={"Retry-After": "5"})

    with patch("src.ingest.github_client.time.sleep") as mock_sleep:
        assert _sleep_for_rate_limit(response) is True
        mock_sleep.assert_called_once_with(6)


def test_sleep_for_rate_limit_waits_until_reset_when_exhausted():
    response = Mock(
        status_code=429,
        headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1000"},
    )

    with (
        patch("src.ingest.github_client.time.sleep") as mock_sleep,
        patch("src.ingest.github_client.time.time", return_value=990),
    ):
        assert _sleep_for_rate_limit(response) is True
        mock_sleep.assert_called_once_with(11)  # (1000 - 990) + 1


def test_sleep_for_rate_limit_false_alarm_403_with_no_rate_limit_headers():
    response = Mock(status_code=403, headers={})

    with patch("src.ingest.github_client.time.sleep") as mock_sleep:
        assert _sleep_for_rate_limit(response) is False
        mock_sleep.assert_not_called()


def test_paginated_get_retries_and_follows_pagination():
    rate_limited = Mock(status_code=403, headers={"Retry-After": "0"})
    page1 = Mock(links={"next": {"url": "https://api.github.com/page2"}})
    page1.json.return_value = [{"id": 1}]
    page2 = Mock(links={})
    page2.json.return_value = [{"id": 2}]

    session = Mock()
    session.get.side_effect = [rate_limited, page1, page2]

    with patch("src.ingest.github_client.time.sleep"):
        items = list(paginated_get(session, "https://api.github.com/page1"))

    assert items == [{"id": 1}, {"id": 2}]
    assert session.get.call_count == 3
