"""
Supabase Auth helper — lightweight client using httpx (no heavy SDK needed).
Calls the Supabase Auth REST API directly for sign-up, sign-in, and sign-out.
"""

import os
import httpx

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.environ.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_ANON_KEY. "
        "Set them in your .env file (local) or Render environment variables."
    )

_AUTH_BASE = f"{SUPABASE_URL}/auth/v1"
_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Content-Type": "application/json",
}


def sign_up(email: str, password: str) -> dict:
    """Create a new user account. Returns the parsed JSON response."""
    resp = httpx.post(
        f"{_AUTH_BASE}/signup",
        headers=_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    data = resp.json()
    if resp.status_code >= 400:
        msg = data.get("error_description") or data.get("msg") or str(data)
        raise Exception(msg)
    return data


def sign_in(email: str, password: str) -> dict:
    """Sign in with email/password. Returns user + session data."""
    resp = httpx.post(
        f"{_AUTH_BASE}/token?grant_type=password",
        headers=_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    data = resp.json()
    if resp.status_code >= 400:
        msg = data.get("error_description") or data.get("msg") or str(data)
        raise Exception(msg)
    return data


def sign_out(access_token: str) -> None:
    """Sign out the current user (invalidates the access token)."""
    httpx.post(
        f"{_AUTH_BASE}/logout",
        headers={**_HEADERS, "Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
