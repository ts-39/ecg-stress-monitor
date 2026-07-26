# Credentials

This directory holds OAuth secrets for Google Calendar integration.
**These files are listed in `.gitignore` and must never be committed.**

## Setup

1. Go to https://console.cloud.google.com/
2. Create (or reuse) a project and enable the Google Calendar API.
3. Create an OAuth 2.0 Client ID (type: Web application).
4. Add `http://localhost:8000/api/calendar/callback` to "Authorized redirect URIs".
5. Download the JSON and place it here as:
   `client_secret_<client_id>.apps.googleusercontent.com.json`

`google_token.json` is generated automatically on first successful OAuth login.
