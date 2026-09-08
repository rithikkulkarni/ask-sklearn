"""Per-IP rate limiting via slowapi, backed by its default in-memory storage --
no extra infrastructure, which is fine for a low-traffic public demo.

The limiter is a module-level singleton so main.py and the endpoint decorator
can both reference it without a circular import.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
