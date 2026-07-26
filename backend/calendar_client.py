from __future__ import annotations

import datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

CREDS_FILE   = Path(__file__).parent / "client_secret_342843476489-7j1arqcoonimv6p3oc5vu6n97i8sok9o.apps.googleusercontent.com.json"
TOKEN_FILE   = Path(__file__).parent / "google_token.json"
SCOPES       = ["https://www.googleapis.com/auth/calendar.readonly"]
REDIRECT_URI = "http://localhost:8000/api/calendar/callback"

# The Flow object holds the PKCE code_verifier after authorization_url() is called.
# Reuse the same instance in handle_callback() so the verifier is not lost.
_pending_flow: Flow | None = None


def get_auth_url() -> str:
    global _pending_flow
    _pending_flow = Flow.from_client_secrets_file(str(CREDS_FILE), scopes=SCOPES)
    _pending_flow.redirect_uri = REDIRECT_URI
    auth_url, _ = _pending_flow.authorization_url(access_type="offline", prompt="consent")
    return auth_url


def handle_callback(code: str) -> bool:
    global _pending_flow
    if _pending_flow is None:
        print("[Calendar] No pending OAuth flow — restart auth")
        return False
    try:
        _pending_flow.fetch_token(code=code)
        TOKEN_FILE.write_text(_pending_flow.credentials.to_json())
        print("[Calendar] OAuth token saved")
        return True
    except Exception as e:
        print(f"[Calendar] OAuth callback error: {e}")
        return False
    finally:
        _pending_flow = None


def _load_creds() -> Credentials | None:
    if not TOKEN_FILE.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
        return creds if creds.valid else None
    except Exception as e:
        print(f"[Calendar] Token load error: {e}")
        return None


def is_connected() -> bool:
    return _load_creds() is not None


def get_today_events() -> list[dict]:
    creds = _load_creds()
    if creds is None:
        return []
    try:
        service = build("calendar", "v3", credentials=creds, cache_discovery=False)
        now_dt    = datetime.datetime.now(datetime.timezone.utc)
        day_start = now_dt.replace(hour=0,  minute=0,  second=0,  microsecond=0)
        day_end   = now_dt.replace(hour=23, minute=59, second=59, microsecond=0)
        result = service.events().list(
            calendarId="primary",
            timeMin=day_start.isoformat(),
            timeMax=day_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = []
        for ev in result.get("items", []):
            start_raw = ev["start"].get("dateTime")
            end_raw   = ev["end"].get("dateTime")
            if start_raw is None:   # skip all-day events
                continue
            events.append({
                "id":    ev["id"],
                "title": ev.get("summary", "(no title)"),
                "start": start_raw,
                "end":   end_raw,
            })
        return events
    except Exception as e:
        print(f"[Calendar] API error: {e}")
        return []


def get_current_event() -> dict | None:
    events = get_today_events()
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    for ev in events:
        try:
            start = datetime.datetime.fromisoformat(ev["start"])
            end   = datetime.datetime.fromisoformat(ev["end"])
            if start.tzinfo is None:
                start = start.replace(tzinfo=datetime.timezone.utc)
            if end.tzinfo is None:
                end = end.replace(tzinfo=datetime.timezone.utc)
            if start <= now_dt <= end:
                return ev
        except Exception:
            continue
    return None
