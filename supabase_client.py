"""
Supabase client — lightweight helpers using httpx (no heavy SDK needed).
Covers Auth (sign-up/in/out) and PostgREST (runs, baselines).
"""

import os
import json
import httpx

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY: str = os.environ.get("SUPABASE_ANON_KEY", "")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_ANON_KEY. "
        "Set them in your .env file (local) or Render environment variables."
    )

_AUTH_BASE = f"{SUPABASE_URL}/auth/v1"
_REST_BASE = f"{SUPABASE_URL}/rest/v1"
_HEADERS = {
    "apikey": SUPABASE_ANON_KEY,
    "Content-Type": "application/json",
}


def _auth_headers(access_token: str) -> dict:
    """Headers for authenticated PostgREST requests (RLS uses the JWT)."""
    return {**_HEADERS, "Authorization": f"Bearer {access_token}"}


# ═══════════════════════════════════════════════════════════════════════════
# Auth
# ═══════════════════════════════════════════════════════════════════════════

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
        headers=_HEADERS | {"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Runs (PostgREST)
# ═══════════════════════════════════════════════════════════════════════════

def save_run(access_token: str, user_id: str, file_name: str,
             protocol: str, chemistry: str, dosage: str,
             summary_json: dict, run_data_json: str) -> dict:
    """Insert a new run. Returns the created row (including its id)."""
    resp = httpx.post(
        f"{_REST_BASE}/runs",
        headers=_auth_headers(access_token) | {"Prefer": "return=representation"},
        json={
            "user_id": user_id,
            "file_name": file_name,
            "protocol": protocol or "",
            "chemistry": chemistry or "",
            "dosage": dosage or "",
            "summary_json": summary_json,
            "run_data": run_data_json,
        },
        timeout=30,
    )
    data = resp.json()
    if resp.status_code >= 400:
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)
    return data[0] if isinstance(data, list) else data


def get_runs_list(access_token: str) -> list[dict]:
    """Fetch all runs for the current user (metadata only, no run_data).
    Returns list sorted by created_at descending (newest first)."""
    resp = httpx.get(
        f"{_REST_BASE}/runs",
        headers=_auth_headers(access_token),
        params={
            "select": "id,file_name,protocol,chemistry,dosage,summary_json,created_at",
            "order": "created_at.desc",
        },
        timeout=15,
    )
    data = resp.json()
    if resp.status_code >= 400:
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)
    return data


def get_run_data(access_token: str, run_id: str) -> str:
    """Fetch the full run_data (DataFrame JSON) for a specific run."""
    resp = httpx.get(
        f"{_REST_BASE}/runs",
        headers=_auth_headers(access_token),
        params={
            "select": "run_data",
            "id": f"eq.{run_id}",
        },
        timeout=15,
    )
    data = resp.json()
    if resp.status_code >= 400:
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)
    if not data:
        raise Exception(f"Run {run_id} not found")
    return data[0]["run_data"]


def delete_run(access_token: str, run_id: str) -> None:
    """Delete a run by id."""
    resp = httpx.delete(
        f"{_REST_BASE}/runs",
        headers=_auth_headers(access_token),
        params={"id": f"eq.{run_id}"},
        timeout=10,
    )
    if resp.status_code >= 400:
        data = resp.json()
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)


# ═══════════════════════════════════════════════════════════════════════════
# Baselines (PostgREST) — upsert by (user_id, protocol)
# ═══════════════════════════════════════════════════════════════════════════

def save_baseline(access_token: str, user_id: str,
                  protocol: str, baseline_json: dict) -> None:
    """Upsert a baseline for a protocol. Overwrites if one already exists."""
    resp = httpx.post(
        f"{_REST_BASE}/baselines",
        headers=_auth_headers(access_token) | {
            "Prefer": "resolution=merge-duplicates",
        },
        params={"on_conflict": "user_id,protocol"},
        json={
            "user_id": user_id,
            "protocol": protocol,
            "baseline_json": baseline_json,
        },
        timeout=10,
    )
    if resp.status_code >= 400:
        data = resp.json()
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)


def get_baseline(access_token: str, protocol: str) -> dict | None:
    """Fetch the baseline for a protocol. Returns baseline_json dict or None."""
    resp = httpx.get(
        f"{_REST_BASE}/baselines",
        headers=_auth_headers(access_token),
        params={
            "select": "baseline_json",
            "protocol": f"eq.{protocol}",
        },
        timeout=10,
    )
    data = resp.json()
    if resp.status_code >= 400:
        return None
    if not data:
        return None
    return data[0]["baseline_json"]


# ═══════════════════════════════════════════════════════════════════════════
# User preferences (PostgREST) — upsert by user_id
# ═══════════════════════════════════════════════════════════════════════════

def save_preferences(access_token: str, user_id: str,
                     prefs: dict) -> None:
    """Upsert user preferences. Overwrites if row already exists."""
    resp = httpx.post(
        f"{_REST_BASE}/user_preferences",
        headers=_auth_headers(access_token) | {
            "Prefer": "resolution=merge-duplicates",
        },
        params={"on_conflict": "user_id"},
        json={
            "user_id": user_id,
            "preferences": prefs,
        },
        timeout=10,
    )
    if resp.status_code >= 400:
        data = resp.json()
        msg = data.get("message") or data.get("msg") or str(data)
        raise Exception(msg)


def get_preferences(access_token: str) -> dict | None:
    """Fetch preferences for the current user. Returns dict or None."""
    resp = httpx.get(
        f"{_REST_BASE}/user_preferences",
        headers=_auth_headers(access_token),
        params={"select": "preferences"},
        timeout=10,
    )
    data = resp.json()
    if resp.status_code >= 400:
        return None
    if not data:
        return None
    return data[0]["preferences"]
