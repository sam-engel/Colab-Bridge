#!/usr/bin/env python3
"""
colab_bridge/direct_kernel.py
=================================
Provision a Google Colab runtime programmatically using the same undocumented
internal API that the VS Code Colab extension uses.  No browser automation.

Auth: standard Google OAuth 2.0 with OIDC scopes (VS Code extension's own client).
      First run:  colab --auth
      Credentials stored in colab_bridge/.direct_kernel_creds.json (gitignored).

Features:
  1. Assign any runtime type (CPU, T4, L4, A100, H100, high-RAM variants)
  2. List all active runtimes; unassign any by endpoint name
  3. Submit code (foreground or detached via --no-stream; survives CLI exit)
  4. File-backed job registry — list, replay, follow, cancel from any terminal
  5. Restart kernel
  6. Graceful interrupt; force-interrupt / kernel kill-and-recreate even if frozen
  7. Per-job event logs persisted to colab_bridge/.direct_kernel_jobs/

Usage:
  colab --auth
  colab --list
  colab --unassign m-s-xxx-use1b0-yyy
  colab --test-cpu
  colab -c "print(1+1)"
  colab --accelerator A100 -f experiments/train.py
  colab --no-stream -c "long_script()"
  colab --keepalive --accelerator A100 -f train.py
  colab --jobs           # list every job ever submitted
  colab --latest         # follow the newest job to completion
  colab --cancel JID     # graceful interrupt

Discovered by reverse-engineering ~/.vscode/extensions/google.colab-0.8.0/out/extension.js.
Key: GET /tun/m/assign → xsrf token; POST → runtimeProxyInfo.{url, token, tokenExpiresInSeconds}
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import shutil
import socket
import ssl
import struct
import sys
import threading
import time
import traceback
import urllib.parse
import uuid
from pathlib import Path
from queue import Queue
from typing import Iterator

import requests

# ── IPv4-preferred DNS ────────────────────────────────────────────────────────
# When Tailscale (or some VPNs) is active on macOS, the IPv6 path to Google
# APIs can hang for 15-20s before falling back, while the IPv4 path responds
# in <100ms.  Re-order getaddrinfo results so urllib3 tries IPv4 first; if v4
# fails it still tries v6, so this is a no-op on networks where v6 works.
_orig_getaddrinfo = socket.getaddrinfo
def _v4_first_getaddrinfo(host, port, *args, **kwargs):
    res = _orig_getaddrinfo(host, port, *args, **kwargs)
    return sorted(res, key=lambda r: 0 if r[0] == socket.AF_INET else 1)
socket.getaddrinfo = _v4_first_getaddrinfo

# ── paths ──────────────────────────────────────────────────────────────────────
BRIDGE_DIR     = Path(__file__).parent
_CREDS_FILE    = BRIDGE_DIR / ".direct_kernel_creds.json"
_NOTEBOOK_FILE = BRIDGE_DIR / ".direct_kernel_notebook_id"
_JOBS_DIR      = BRIDGE_DIR / ".direct_kernel_jobs"
_ENV_FILE      = BRIDGE_DIR / ".env"

# ── Colab API constants (from VS Code extension source) ────────────────────────
COLAB_DOMAIN      = "https://colab.research.google.com"
COLAB_GAPI_DOMAIN = "https://colab.pa.googleapis.com"
ASSIGN_PATH       = "/tun/m/assign"

# VS Code Colab extension OAuth client (kr.ClientId / kr.ClientNotSoSecret in extension.js)
_CLIENT_ID     = "1014160490159-cvot3bea7tgkp72a4m29h20d9ddo6bne.apps.googleusercontent.com"
_CLIENT_SECRET = "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"

# OIDC + Colab scopes required by /tun/m/assign
_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]

# Accelerator short name → (accelerator query value, variant query value)
# CPU uses neither.  Discovered via /v1/user-info eligibleAccelerators.
_ACCEL_MAP: dict[str, tuple[str | None, str | None]] = {
    "CPU":  (None,   None),
    "T4":   ("T4",   "GPU"),
    "L4":   ("L4",   "GPU"),
    "A100": ("A100", "GPU"),
    "H100": ("H100", "GPU"),
    "G4":   ("G4",   "GPU"),
    "V5E1": ("V5E1", "TPU"),
    "V6E1": ("V6E1", "TPU"),
}


# ── progress spinner ───────────────────────────────────────────────────────────

_SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

def _spin_thread(label: str, stop: threading.Event) -> None:
    t0 = time.time()
    i  = 0
    while not stop.wait(0.1):
        sys.stderr.write(
            f"\r{_SPINNER_CHARS[i % len(_SPINNER_CHARS)]} {label}… {time.time()-t0:.0f}s"
        )
        sys.stderr.flush()
        i += 1


@contextlib.contextmanager
def _spin(label: str):
    """Show a stderr spinner with elapsed time while the block runs (TTY only)."""
    if not sys.stderr.isatty():
        yield
        return
    stop = threading.Event()
    t = threading.Thread(target=_spin_thread, args=(label, stop), daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        sys.stderr.write("\r\033[K")   # erase the spinner line
        sys.stderr.flush()


# ── XSSI stripping ─────────────────────────────────────────────────────────────

def _xssi_json(text: str) -> dict:
    """Strip Google's XSSI prefix  )]}'  before parsing JSON."""
    if text.startswith(")]}'"):
        text = text[4:].lstrip()
    return json.loads(text)


# ── OAuth ──────────────────────────────────────────────────────────────────────

def do_auth() -> dict:
    """Run the OAuth installed-app flow; save and return credentials."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        raise ImportError("pip install google-auth-oauthlib google-auth")

    import warnings
    warnings.filterwarnings("ignore", ".*Scope.*has changed.*")

    client_config = {
        "installed": {
            "client_id":     _CLIENT_ID,
            "client_secret": _CLIENT_SECRET,
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
        }
    }
    flow  = InstalledAppFlow.from_client_config(client_config, scopes=_SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    stored = {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "client_id":     _CLIENT_ID,
        "client_secret": _CLIENT_SECRET,
    }
    _CREDS_FILE.write_text(json.dumps(stored, indent=2))
    print(f"[direct_kernel] Credentials saved → {_CREDS_FILE.name}", flush=True)
    return stored


def _refresh_token(creds: dict) -> dict:
    """Exchange refresh_token for a new access token; write expires_at to creds file."""
    r = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "client_id":     creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type":    "refresh_token",
        },
        timeout=15,
    )
    r.raise_for_status()
    data  = r.json()
    creds = {
        **creds,
        "token":      data["access_token"],
        "expires_at": time.time() + data.get("expires_in", 3600),
    }
    _CREDS_FILE.write_text(json.dumps(creds, indent=2))
    return creds


def _get_access_token() -> str:
    """Return a valid access token, refreshing locally without a network check."""
    if not _CREDS_FILE.exists():
        raise FileNotFoundError(
            "No credentials.  Run first:\n"
            "  python3 colab_bridge/direct_kernel.py --auth"
        )
    creds = json.loads(_CREDS_FILE.read_text())
    # Use stored expiry — no tokeninfo round-trip needed
    if creds.get("expires_at", 0) > time.time() + 300:
        return creds["token"]
    with _spin("refreshing token"):
        return _refresh_token(creds)["token"]


def make_session() -> requests.Session:
    """Build a requests.Session with a valid OAuth Bearer token + VS Code headers."""
    token = _get_access_token()
    sess  = requests.Session()
    sess.headers.update({
        "Authorization":                     f"Bearer {token}",
        "X-Colab-Client-Agent":              "vscode",
        "X-Colab-VS-Code-App-Name":          "Visual Studio Code",
        "X-Colab-VS-Code-Extension-Version": "0.8.0",
    })
    return sess


def _ensure_fresh_token(sess: requests.Session) -> None:
    """Make sure the session's Authorization header carries a valid OAuth
    token.  Two cases:

    1. Disk creds are about to expire (within 5 min) → refresh on disk and
       update the session header.
    2. Disk creds are still valid but the session header holds an OLDER
       token than what's on disk (because another process refreshed the
       token between our session creation and now) → just sync the header
       from disk without a network call.

    Case (2) is critical for long-lived background threads (reaper's
    gRPC keep-alive, ws-keepalive, etc.) that share `_CREDS_FILE` with
    foreground CLI invocations.  Without this sync, the thread silently
    keeps using its initial-init token until that token's own expiry,
    then 401s forever even though disk has a fresh token.
    """
    creds = json.loads(_CREDS_FILE.read_text())
    if creds.get("expires_at", 0) <= time.time() + 300:
        new_token = _get_access_token()
        sess.headers["Authorization"] = f"Bearer {new_token}"
        return
    disk_token = creds.get("token")
    if disk_token and sess.headers.get("Authorization") != f"Bearer {disk_token}":
        sess.headers["Authorization"] = f"Bearer {disk_token}"


# ── notebook ───────────────────────────────────────────────────────────────────

def _notebook_hash(file_id: str) -> str:
    """Compute the nbh parameter (from VS Code extension function Mg)."""
    return file_id.replace("-", "_") + "." * (44 - len(file_id))


# Minimal valid .ipynb body.  Without a body Colab's UI shows
# "The file has been corrupted or is not a valid notebook file."
_NOTEBOOK_TEMPLATE: dict = {
    "cells":         [],
    "metadata":      {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat":      4,
    "nbformat_minor": 5,
}


def _drive_multipart_upload(sess: requests.Session, metadata: dict, body: dict) -> dict:
    """Create a Drive file with both metadata + content in one multipart request."""
    boundary = "----direct_kernel_boundary"
    payload = (
        f"--{boundary}\r\n"
        f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{json.dumps(metadata)}\r\n"
        f"--{boundary}\r\n"
        f"Content-Type: application/vnd.google.colaboratory\r\n\r\n"
        f"{json.dumps(body)}\r\n"
        f"--{boundary}--"
    ).encode()
    r = sess.post(
        "https://www.googleapis.com/upload/drive/v3/files",
        params={"uploadType": "multipart"},
        headers={"Content-Type": f"multipart/related; boundary={boundary}"},
        data=payload,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def repair_notebook(sess: requests.Session, fid: str) -> None:
    """Overwrite a Drive file with a minimal valid .ipynb body so Colab's UI
    can open it.  Use when the saved notebook was created by an old version of
    this script (metadata-only, no body) and shows up as 'corrupted'."""
    r = sess.patch(
        f"https://www.googleapis.com/upload/drive/v3/files/{fid}",
        params={"uploadType": "media"},
        headers={"Content-Type": "application/vnd.google.colaboratory"},
        data=json.dumps(_NOTEBOOK_TEMPLATE).encode(),
        timeout=20,
    )
    r.raise_for_status()


def get_or_create_notebook(sess: requests.Session) -> str:
    """Return a Drive file ID for a reusable throw-away Colab notebook."""
    if _NOTEBOOK_FILE.exists():
        fid = _NOTEBOOK_FILE.read_text().strip()
        r = sess.get(
            f"https://www.googleapis.com/drive/v3/files/{fid}",
            params={"fields": "id,trashed,size"},
            timeout=10,
        )
        if r.status_code == 200 and not r.json().get("trashed"):
            # Heal old notebooks that were created without a body.
            try:
                if int(r.json().get("size", 0)) == 0:
                    print("[direct_kernel] Repairing empty saved notebook…", flush=True)
                    repair_notebook(sess, fid)
            except Exception:
                pass
            return fid
        print("[direct_kernel] Saved notebook gone — creating a new one.", flush=True)

    print("[direct_kernel] Creating Colab notebook on Drive…", flush=True)
    try:
        data = _drive_multipart_upload(
            sess,
            metadata={
                "name":     "direct_kernel_runtime",
                "mimeType": "application/vnd.google.colaboratory",
            },
            body=_NOTEBOOK_TEMPLATE,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (401, 403):
            raise PermissionError(
                f"Drive API {e.response.status_code}.  Run --auth to re-authenticate."
            ) from e
        raise
    fid = data["id"]
    _NOTEBOOK_FILE.write_text(fid)
    print(f"[direct_kernel] Created notebook {fid}", flush=True)
    return fid


# ── local secrets (.env) ──────────────────────────────────────────────────────
#
# Colab Secrets are blocked from non-UI clients (see README §"Colab Secrets"),
# so we keep secrets on the user's machine and inject them into each job's
# execute_request as a prelude that sets ``os.environ``.  The prelude is sent
# over TLS but is NOT stored in ``index.json`` — only the user's original code
# is persisted.

def _load_env() -> dict[str, str]:
    """Parse ``colab_bridge/.env``.  Format: ``KEY=VALUE`` per line.

    * Comments (``#``) and blank lines skipped.
    * Single-line values may be plain or wrapped in matching single / double quotes.
    * **Multi-line quoted values are supported**: a value of the form
      ``KEY="…`` opens a block that continues across newlines until a line
      ending in the same un-escaped quote.  Useful for service-account JSON
      keys etc.
    * Lines that don't look like ``KEY=…`` are silently ignored.
    """
    if not _ENV_FILE.exists():
        return {}
    out: dict[str, str] = {}
    lines = _ENV_FILE.read_text().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = raw.partition("=")
        k = k.strip()
        if not k:
            continue
        v = v.strip()
        if v and v[0] in ("'", '"'):
            quote = v[0]
            # Closed on the same line?
            if len(v) >= 2 and v[-1] == quote and v != quote:
                out[k] = v[1:-1].replace("\\" + quote, quote)
                continue
            # Otherwise: multi-line block — accumulate until we see the
            # closing quote at the end of a line (un-escaped).
            buf = [v[1:]]
            while i < len(lines):
                ln = lines[i]; i += 1
                stripped_r = ln.rstrip()
                if stripped_r.endswith(quote) and not stripped_r.endswith("\\" + quote):
                    buf.append(stripped_r[:-1])
                    break
                buf.append(ln)
            out[k] = "\n".join(buf).replace("\\" + quote, quote)
        else:
            out[k] = v
    return out


def _save_env(values: dict[str, str]) -> None:
    """Rewrite ``colab_bridge/.env`` with the given KEY=VALUE pairs.

    Single-line values are quoted only when they contain whitespace or special
    characters.  **Multi-line values are written as multi-line quoted blocks**
    so a service-account JSON (etc.) round-trips through ``_load_env``
    losslessly.  Comments in the original file are NOT preserved.
    """
    lines = []
    for k in sorted(values):
        v = values[k]
        if "\n" in v:
            # Multi-line — pick the quote that doesn't appear in the value if
            # possible, otherwise escape.
            if "'" not in v:
                lines.append(f"{k}='{v}'")
            elif '"' not in v:
                lines.append(f'{k}="{v}"')
            else:
                escaped = v.replace('"', '\\"')
                lines.append(f'{k}="{escaped}"')
        else:
            needs_quote = (
                v == ""
                or any(ch in v for ch in (" ", "\t", '"', "'", "#", "$", "\\"))
            )
            if needs_quote:
                v = '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
            lines.append(f"{k}={v}")
    _ENV_FILE.write_text("\n".join(lines) + "\n")
    try:
        os.chmod(_ENV_FILE, 0o600)   # owner read/write only — secrets file
    except Exception:
        pass


def _build_env_prelude(values: dict[str, str]) -> str:
    """Return Python source that updates os.environ with the given values.

    The prelude wraps in a try/except that surfaces only the exception type,
    never values, so a malformed entry can't leak via traceback.  Uses
    ``json.dumps`` so the dict literal is safe regardless of value contents.
    """
    if not values:
        return ""
    payload = json.dumps(values)
    return (
        "try:\n"
        "    import os as _dk_os\n"
        f"    _dk_os.environ.update({payload})\n"
        "    del _dk_os\n"
        "except Exception as _dk_e:\n"
        "    import sys as _dk_sys\n"
        "    print(f'[direct_kernel] env injection error: {type(_dk_e).__name__}',\n"
        "          file=_dk_sys.stderr)\n"
        "    del _dk_sys, _dk_e\n"
    )


def _resolve_env_for_job(inject: bool | list[str] | None) -> dict[str, str]:
    """Return the (filtered) {name: value} dict to inject into a job.

    ``inject``:
      * ``True`` or ``None`` → inject everything in ``.env``.
      * ``False``            → inject nothing.
      * list of names        → inject only those (silently skip missing).
    """
    if inject is False:
        return {}
    env = _load_env()
    if inject is None or inject is True:
        return {k: v for k, v in env.items() if v != ""}
    return {k: env[k] for k in inject if env.get(k)}


# ── runtime metadata (idle-timeout, assigned-at, …) ───────────────────────────

_RUNTIME_META_FILE = _JOBS_DIR / "runtimes.json"
_DEFAULT_IDLE_TIMEOUT_MIN = 30

# Strip CSI escapes (color/SGR + cursor + erase + private modes).  Used by
# the live dashboard's commit path to drop ANSI-only lines (e.g. tqdm's
# inter-bar `\x1b[K`) so they don't render as blank rows in the follow view.
_ANSI_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _runtime_meta_all() -> dict[str, dict]:
    if not _RUNTIME_META_FILE.exists():
        return {}
    try:
        return json.loads(_RUNTIME_META_FILE.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def _runtime_meta(endpoint: str) -> dict | None:
    return _runtime_meta_all().get(endpoint)


def _runtime_meta_lock():
    """fcntl.flock on the runtime metadata file."""
    import fcntl
    _JOBS_DIR.mkdir(exist_ok=True)
    if not _RUNTIME_META_FILE.exists():
        _RUNTIME_META_FILE.write_text("{}")
    f = _RUNTIME_META_FILE.open("r+")
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    return f


def _set_runtime_meta(endpoint: str, **fields) -> dict:
    f = _runtime_meta_lock()
    try:
        f.seek(0)
        data = json.loads(f.read() or "{}")
        rec  = data.setdefault(endpoint, {})
        rec.update(fields)
        f.seek(0); f.truncate()
        json.dump(data, f, indent=2)
        return rec
    finally:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def _remove_runtime_meta(endpoint: str) -> None:
    f = _runtime_meta_lock()
    try:
        f.seek(0)
        data = json.loads(f.read() or "{}")
        if endpoint in data:
            del data[endpoint]
        f.seek(0); f.truncate()
        json.dump(data, f, indent=2)
    finally:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


# ── runtime management ─────────────────────────────────────────────────────────

# ── event log (alerts) ────────────────────────────────────────────────────────

_EVENTS_LOG       = _JOBS_DIR / "events.jsonl"
_EVENTS_KEEP      = 10000      # rotate after this many lines
_EVENT_TYPES_HELP = (
    "job_queued, job_started, job_done, job_error, job_cancelled, job_timeout, "
    "runtime_assigned, runtime_released"
)


def _emit_event(event_type: str, **fields) -> None:
    """Append one event line to events.jsonl.  Never raises."""
    try:
        _JOBS_DIR.mkdir(exist_ok=True)
        rec = {"ts": time.time(), "type": event_type, **fields}
        with _EVENTS_LOG.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        # Cheap rotation: count lines occasionally and truncate if huge.
        try:
            stat = _EVENTS_LOG.stat()
            # Only check on every ~100th call (when size > ~1MB).
            if stat.st_size > 2_000_000:
                lines = _EVENTS_LOG.read_text().splitlines()
                if len(lines) > _EVENTS_KEEP:
                    _EVENTS_LOG.write_text("\n".join(lines[-_EVENTS_KEEP:]) + "\n")
        except Exception:
            pass
    except Exception:
        pass


def _watch_events(
    types:    set[str] | None = None,
    jid:      str | None = None,
    endpoint: str | None = None,
    once:     bool = False,
    from_start: bool = False,
) -> int:
    """Tail events.jsonl with optional filtering.  Prints one JSON line per
    matching event.  ``once=True`` exits 0 after the first match.

    Returns: 0 on clean exit, non-zero only on unrecoverable failure.
    Designed to be backgrounded by an agent / wrapped by ``notify-send``.
    """
    _JOBS_DIR.mkdir(exist_ok=True)
    if not _EVENTS_LOG.exists():
        _EVENTS_LOG.write_text("")
    pos = 0 if from_start else _EVENTS_LOG.stat().st_size
    poll = 0.25
    try:
        while True:
            with _EVENTS_LOG.open() as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for line in chunk.splitlines():
                if not line.strip(): continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if types is not None and ev.get("type") not in types:
                    continue
                if jid is not None and ev.get("jid") != jid:
                    continue
                if endpoint is not None and ev.get("endpoint") != endpoint:
                    continue
                sys.stdout.write(json.dumps(ev) + "\n")
                sys.stdout.flush()
                if once:
                    return 0
            time.sleep(poll)
    except KeyboardInterrupt:
        return 0


def _wait_for_job(jid: str) -> int:
    """Block until the named job leaves queued/running.  Print the final
    status line and exit with 0 on success, 1 on error/cancelled, 2 if the
    job is unknown.

    Implemented as a `_watch_events` filter so it benefits from event-log
    history (works even if the job ended before this command was invoked).
    """
    store = _JobStore()
    rec = store.get_job(jid)
    if rec is None:
        print(f"[direct_kernel] unknown job_id: {jid!r}", file=sys.stderr)
        return 2
    if rec.get("status") not in ("queued", "running"):
        # Already terminal — no event needed.
        ev = {"ts": rec.get("ended", time.time()),
              "type": f"job_{rec['status']}", "jid": jid,
              "endpoint": rec.get("endpoint")}
        sys.stdout.write(json.dumps(ev) + "\n")
        return 0 if rec["status"] == "done" else 1
    return _watch_events(
        types={"job_done", "job_error", "job_cancelled", "job_timeout"},
        jid=jid, once=True,
    )


def _stream_events_human() -> int:
    """Tail events.jsonl forever, printing one human-readable line per event.

    Color-coded by event type (release: red preempted, yellow others;
    job_done green, job_error red, etc.).  Ctrl+C exits cleanly.
    """
    _JOBS_DIR.mkdir(exist_ok=True)
    if not _EVENTS_LOG.exists():
        _EVENTS_LOG.write_text("")

    def fmt(ev: dict) -> tuple[str, int]:
        t   = ev.get("type", "?")
        ts  = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", time.time())))
        jid = ev.get("jid", "")
        ep  = ev.get("endpoint", "")
        ac  = ev.get("accel", "")
        if t == "runtime_assigned":
            extra = " (reused)" if ev.get("reused") else ""
            return f"{ts}  ++  runtime assigned   {ep}  ({ac}, {ev.get('region','')}){extra}", 32
        if t == "runtime_released":
            reason = ev.get("reason", "?")
            color = 91 if reason == "preempted" else 33
            return f"{ts}  ✗   runtime released   {ep}  reason={reason}", color
        if t == "job_queued":
            return f"{ts}  +   job queued         {jid}  on {ep}  ({ac})", 36
        if t == "job_started":
            return f"{ts}  ▶   job started        {jid}  ({ac})", 32
        if t == "job_done":
            return f"{ts}  ✔   job done           {jid}  elapsed={ev.get('elapsed_s','?')}s", 32
        if t == "job_error":
            return f"{ts}  ✖   job errored        {jid}  elapsed={ev.get('elapsed_s','?')}s", 91
        if t == "job_cancelled":
            return f"{ts}  ✗   job cancelled      {jid}  elapsed={ev.get('elapsed_s','?')}s", 33
        if t == "job_timeout":
            return f"{ts}  ⏱   job timed out      {jid}  elapsed={ev.get('elapsed_s','?')}s", 33
        return f"{ts}  ·   {t}: {ev}", 36

    print(f"\033[36m●  events stream — Ctrl+C to exit  ({_EVENTS_LOG})\033[0m")
    pos = _EVENTS_LOG.stat().st_size
    try:
        while True:
            with _EVENTS_LOG.open() as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for line in chunk.splitlines():
                if not line.strip(): continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                msg, color = fmt(ev)
                print(f"\033[{color}m{msg}\033[0m", flush=True)
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n\033[36m●  events stream stopped.\033[0m")
        return 0


# ── `/` command palette (used by --live) ──────────────────────────────────────
# (name, signature, description) — canonical names only; aliases live in the
# dispatcher.  Order here is the order shown in /help and in autocomplete.
_PALETTE_CMDS: list[tuple[str, str, str]] = [
    ("help",     "",                          "show this list"),
    ("cancel",   "<jid>",                     "graceful interrupt of a running job"),
    ("release",  "<runtime>",                 "unassign a runtime  (alias /unassign)"),
    ("timeout",  "<runtime> <min>",           "set idle-timeout (0 = disable)"),
    ("desc",     "<rt|jid> <text>",           "set description on a runtime or job"),
    ("assign",   "<accel> [desc]",            "spawn `colab --assign` in the background"),
    ("run",      "<runtime> <code>",          "spawn detached `--no-stream -c` job (alias /submit)"),
    ("reattach", "<jid>",                     "reattach to a job whose watcher died (capture new output to events.jsonl)"),
    ("jobs",     "[all]",                     "switch to jobs view; pass `all` to include terminal-status jobs"),
    ("runtimes", "[all]",                     "switch to instances view; pass `all` to include released ones (alias /instances /list)"),
    ("instances","[all]",                     "alias of /runtimes"),
    ("list",     "[all]",                     "alias of /runtimes"),
    ("cost",     "",                          "switch to cost view"),
    ("follow",   "[runtime]",                 "switch to follow view (optionally scoped to a runtime SH letter, e.g. /follow a)"),
    ("latest",   "",                          "switch to follow view scoped to the newest running job's runtime"),
    ("events",   "",                          "switch to events view"),
    ("overview", "",                          "switch to overview"),
    ("status",   "<jid>",                     "show full metadata + status of a job"),
    ("balance",  "",                          "fetch current credit balance"),
    ("event",    "<type> [k=v ...]",          "emit a contrived event into events.jsonl (for waking up `--watch --type X --once` consumers)"),
    ("autocomplete","[on|off]",                "toggle the palette's autocomplete suggestion list (default on; off = no sigline / no list / no ghost text)"),
    ("reload",   "",                          "tear down and re-exec the dashboard (clears stuck `⚠ network` banner + any cached state)"),
    ("env",      "list|set NAME=VAL|rm NAME|show NAME", "manage colab_bridge/.env"),
    ("log",      "[n]",                       "tail reaper.log (default 20 lines)"),
    ("keepalive","[n]",                       "tail ws_coverage.log (gRPC keepalive + WS probe + resources poll, default 20)"),
    ("clear",    "",                          "clear recent-output + events buffers"),
    ("refresh",  "",                          "force an immediate data refresh"),
    ("notebook-url", "",                      "print the bridge's Drive notebook URL"),
    ("quit",     "",                          "exit dashboard (alias /q /exit)"),
]


def _common_str_prefix(strs: list[str]) -> str:
    if not strs: return ""
    s1, s2 = min(strs), max(strs)
    for i, c in enumerate(s1):
        if i >= len(s2) or c != s2[i]:
            return s1[:i]
    return s1


def _live_dashboard() -> int:
    """Full-screen live dashboard.  Updates every ~1s.  Keyboard navigation:
    overview / runtimes / jobs / cost / follow-output / events.  Press `/`
    to open the inline command palette (see _PALETTE_CMDS).  Ctrl+C or `q`
    exits, restoring the cursor + terminal mode.
    """
    from collections import deque
    import select, termios, tty

    HIDE = "\033[?25l"; SHOW = "\033[?25h"
    HOME = "\033[H";    CLEAR_BELOW = "\033[J"
    ALT_ON = "\033[?1049h"     # enter alternate screen — scroll-safe like top
    ALT_OFF = "\033[?1049l"    # leave alternate screen — restores scrollback
    WRAP_OFF = "\033[?7l"      # disable autowrap; we hard-clip lines ourselves
    WRAP_ON = "\033[?7h"
    RECENT_MAX  = 500          # full tail buffer for `f` view; overview slices last 2
    EVENTS_TAIL = 500          # full tail buffer for `e` view; overview slices last 2

    isatty = sys.stdin.isatty()
    fd     = sys.stdin.fileno() if isatty else None
    old_term = termios.tcgetattr(fd) if isatty else None
    # If a previous --live was killed without restoring, old_term might already
    # be in raw mode.  Force the basic sane bits on so the restore at exit
    # always leaves a usable terminal.
    if old_term is not None:
        old_term[1] |= termios.OPOST | termios.ONLCR
        old_term[3] |= termios.ECHO | termios.ICANON | termios.ISIG

    # Bytes carried over from the previous _read_key call that were drained
    # from stdin but not consumed by the current keystroke (e.g. the second
    # `\x1b[B` of a back-to-back wheel-down pair).  Without this buffer, a
    # rapid burst of arrow / wheel events would lose all but the first.
    _pending_buf: list[str] = []

    def _read_key(timeout: float = 0.0) -> str | None:
        """Drain ALL pending stdin chars, skip terminal escape sequences
        (arrow keys, scroll-wheel mouse-tracking, etc.), and return the first
        real keystroke or a normalized navigation token.  A lone ESC tap
        (no follow-up bytes within ~20 ms) is returned as ``"\\x1b"``.
        Returns None if only unrecognized escape sequences (or nothing) was
        pending.

        Why: `\\x1b[<64;…M` (mouse scroll), `\\x1b[A` (arrow up), etc., all
        start with ESC.  If we treated each char individually we'd hit ESC
        first and bounce the view.  Instead we read everything currently
        available and skip past full escape sequences — but distinguish a
        bare ESC keypress from the introducer of a CSI/SS3 sequence by
        peeking briefly for a follow-up byte.
        """
        if not isatty: return None
        # Start with anything left over from a previous call.  Crucial for
        # back-to-back arrow / wheel events: macOS scroll-wheel typically
        # sends 3 lines per "click" as 3 separate `\x1b[A`/`[B` sequences,
        # all queued in stdin.  A naive parse-and-return of one drops the
        # other two; we'd see scroll move one step and then "stick".
        pending: list[str] = list(_pending_buf)
        _pending_buf.clear()
        # First read can wait for `timeout`; subsequent reads must be 0
        # (we only want what's currently buffered).  If we already have
        # leftover bytes, no need to block at all.
        first = (not pending)
        while True:
            try:
                r, _, _ = select.select([sys.stdin], [], [], timeout if first else 0)
                if not r: break
                # Use os.read on the raw fd: `sys.stdin.read(1)` goes
                # through Python's BufferedReader which can hold bytes
                # back waiting for a buffer fill — that defeats the whole
                # point of cbreak mode and made single arrow keypresses
                # intermittent (auto-repeat / sustained scroll worked
                # because enough bytes accumulated to satisfy the buffer).
                b = os.read(fd, 1)
                if not b: break
                pending.append(b.decode("utf-8", errors="replace"))
                first = False
            except Exception:
                break
        # Walk the pending bytes, skipping escape sequences.  Every return
        # path saves pending[i:] to `_pending_buf` so a subsequent call
        # picks up where this one left off — without that, back-to-back
        # arrow / wheel events get truncated (wheel-down click typically
        # produces 3 events; we'd parse one and lose two).
        def _emit(token: str | None, advance_to: int) -> str | None:
            _pending_buf[:] = pending[advance_to:]
            return token

        i = 0
        while i < len(pending):
            c = pending[i]
            if c == "\x1b":
                # Lone ESC vs. CSI / SS3 / mouse-tracking introducer.
                # If we don't yet have the next byte, wait briefly for
                # it.  10 ms is plenty for local terminals (kernel
                # delivers all bytes of an arrow sequence in microseconds
                # when in cbreak mode); we exit the WAIT immediately on
                # final-byte arrival anyway.  ESC is a no-op at the
                # dashboard level, so a misdetected arrow → bare-ESC race
                # is silent rather than disruptive.
                if i + 1 >= len(pending):
                    deadline = time.time() + 0.01
                    while True:
                        if len(pending) > i + 1:
                            last = pending[-1]
                            if last.isalpha() or last in ("~", "m"):
                                break    # full sequence in hand
                        tmo = max(0.0, deadline - time.time())
                        if tmo <= 0: break
                        try:
                            r2, _, _ = select.select([sys.stdin], [], [], tmo)
                            if not r2: break
                            b = os.read(fd, 1)
                            if not b: break
                            pending.append(b.decode("utf-8", errors="replace"))
                        except Exception:
                            break
                if i + 1 >= len(pending):
                    return _emit("\x1b", i + 1)   # bare ESC keypress
                # CSI / SS3 / mouse / etc.  Normalize the handful of
                # navigation sequences the live dashboard cares about, and
                # skip everything else (including focus-in / focus-out
                # reports `\x1b[I` / `\x1b[O` that some terminals send when
                # the window gains/loses focus — those used to confuse the
                # dispatch and effectively dismiss the panel).
                j = i + 1
                prefix = pending[j] if j < len(pending) else ""
                if prefix in ("[", "O"):
                    j += 1
                body: list[str] = []
                while j < len(pending) and not (pending[j].isalpha() or pending[j] in ("~", "m")):
                    body.append(pending[j]); j += 1
                final = pending[j] if j < len(pending) else ""
                if j < len(pending):
                    j += 1   # consume the final byte
                seq = "".join(body) + final
                tok: str | None = None
                if prefix == "[":
                    if seq == "A": tok = "__UP__"
                    elif seq == "B": tok = "__DOWN__"
                    elif seq == "5~": tok = "__PGUP__"
                    elif seq == "6~": tok = "__PGDN__"
                    elif seq == "H": tok = "__HOME__"
                    elif seq == "F": tok = "__END__"
                    elif seq.startswith("<64;") and final in ("M", "m"):
                        tok = "__WHEEL_UP__"
                    elif seq.startswith("<65;") and final in ("M", "m"):
                        tok = "__WHEEL_DOWN__"
                elif prefix == "O":
                    if seq == "A": tok = "__UP__"
                    elif seq == "B": tok = "__DOWN__"
                    elif seq == "H": tok = "__HOME__"
                    elif seq == "F": tok = "__END__"
                if tok is not None:
                    return _emit(tok, j)
                # X10 mouse legacy: `\x1b[M` is followed by 3 raw data
                # bytes (button, x+32, y+32) which are NOT escape
                # sequences — skip them too or they'll be returned as
                # plain key chars (e.g. SPACE → overview, accidentally
                # closing /help on every wheel click).
                if prefix == "[" and seq == "M" and j + 3 <= len(pending):
                    i = j + 3
                    continue
                # Unrecognized escape (focus reports, mouse motion in non-
                # SGR mode, etc.) — skip this sequence and keep walking
                # so a real keystroke later in the buffer still surfaces.
                i = j
                continue
            return _emit(c, i + 1)
        return _emit(None, len(pending))

    if isatty:
        # Disable canonical mode (one char at a time) and echo (no key spam).
        # Explicitly enable OPOST + ONLCR so '\n' becomes CR+LF and lines
        # line up — `tcgetattr` could be returning a raw-mode terminal if a
        # previous --live run was killed before restoring (this manifested as
        # a staircase render).  Forcing the bits on makes us self-healing.
        new_term = termios.tcgetattr(fd)
        new_term[1] |= termios.OPOST | termios.ONLCR
        new_term[3] &= ~(termios.ICANON | termios.ECHO)
        # With ICANON off, VMIN / VTIME control read() blocking.  Force
        # VMIN=0, VTIME=0 so `sys.stdin.read(1)` returns whatever is
        # immediately available without waiting — without this, an
        # inherited VMIN of 1 makes the second-and-subsequent reads of
        # an arrow-key sequence (`[`, `A`/`B`) block, defeating our
        # `select(timeout=0)` polling and dropping intermittent arrows.
        new_term[6] = list(new_term[6])
        new_term[6][termios.VMIN]  = 0
        new_term[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, new_term)
    # Enter alternate screen so the dashboard is scroll-isolated from the
    # user's shell scrollback (just like top/htop/vim/less).  On exit we
    # leave alt screen and the scrollback is exactly as it was before.
    sys.stdout.write(ALT_ON + WRAP_OFF + HIDE); sys.stdout.flush()
    try:
        store = _JobStore()
        sess  = make_session()
        last_balance: float | None = None
        last_balance_ts: float = 0.0
        # Tail-state for the per-open-job event files.
        recent_output: deque = deque(maxlen=RECENT_MAX)
        last_pos:      dict[str, int] = {}    # jid → last-seen byte offset
        # Tail-state for events.jsonl — seed the buffer with the last
        # EVENTS_TAIL historical events so the section isn't empty on launch.
        events_pos: int = 0
        events_buf: deque = deque(maxlen=EVENTS_TAIL)
        if _EVENTS_LOG.exists():
            try:
                lines = _EVENTS_LOG.read_text().splitlines()
                for ln in lines[-EVENTS_TAIL:]:
                    if not ln.strip(): continue
                    try: events_buf.append(json.loads(ln))
                    except Exception: pass
                events_pos = _EVENTS_LOG.stat().st_size
            except Exception:
                pass
        # View state
        view = "overview"   # "overview" | "runtimes" | "jobs" | "cost" | "follow" | "events"
        follow_filter: str | None = None    # set by `/follow QUERY` to scope render_follow to one runtime
        view_all: bool = False              # toggled by `a` key on jobs/runtimes view; broadens the listing

        # Scroll state per scrollable view.  scroll_offsets[v] is the index of
        # the first body item visible (header lines aren't scrolled).  at_tail
        # tracks whether the view is "live-tailing" — when True the offset is
        # snapped to the bottom on each render so new entries auto-appear.
        # Up/Down arrows manipulate these.  Tail resumes when the user scrolls
        # back to the bottom on a tail-style view.
        scroll_offsets = {"runtimes": 0, "jobs": 0, "cost": 0, "follow": 0, "events": 0}
        at_tail        = {"runtimes": False, "jobs": False, "cost": False,
                          "follow": True, "events": True}
        scrollable_views = set(scroll_offsets.keys())
        # ── `/` command palette state ──────────────────────────────────────
        cmd_input: str | None = None         # current buffer (None = not editing)
        palette_show_all: bool = False       # True after Tab on empty buffer
        palette_sel: int = 0                 # selected suggestion row (Tab fills sugg[palette_sel])
        palette_autocomplete: bool = True    # `/autocomplete off` hides sigline + suggestion list (Tab still works)
        last_cmd_lines: list[str] = []       # output of the most recent command
        last_cmd_until: float = 0.0          # timestamp until which to display it
        panel_scroll: int = 0                # scroll offset into last_cmd_lines (Up/Down arrows when panel visible)
        last_render_at: float = 0.0          # for ~30 fps render throttle on scroll bursts

        # ── network state — fetched in background threads ──────────────────
        # `list_runtimes` and `_account_balance` are HTTPS calls with multi-
        # second timeouts (15 s).  Calling them synchronously per dashboard
        # tick freezes the entire UI when wifi drops.  Instead we kick off a
        # fetch in a daemon thread and render the last-known-good result;
        # the dashboard stays responsive while the request hangs.
        net_lock = threading.Lock()
        net_state: dict = {
            "rts":           [],     # last successful list_runtimes() result
            "rts_at":        0.0,    # wall-clock of last successful fetch
            "rts_inflight":  False,
            "rts_err":       None,   # last exception class name (if recent failure)
            "bal":           None,   # last successful _account_balance()
            "bal_at":        0.0,
            "bal_inflight":  False,
        }

        def _async_fetch_rts() -> None:
            with net_lock:
                if net_state["rts_inflight"]:
                    return
                net_state["rts_inflight"] = True
            def _worker():
                try:
                    new = list_runtimes(sess)
                    _heal_stale_released(new)
                    _stamp_missing_letters(new)
                    with net_lock:
                        net_state["rts"]     = new
                        net_state["rts_at"]  = time.time()
                        net_state["rts_err"] = None
                except Exception as exc:
                    with net_lock:
                        net_state["rts_err"] = type(exc).__name__
                finally:
                    with net_lock:
                        net_state["rts_inflight"] = False
            threading.Thread(target=_worker, daemon=True).start()

        def _async_fetch_balance() -> None:
            with net_lock:
                if net_state["bal_inflight"]:
                    return
                net_state["bal_inflight"] = True
            def _worker():
                try:
                    b = _account_balance(sess)
                    if b is not None:
                        try: _record_balance(b)
                        except Exception: pass
                    with net_lock:
                        net_state["bal"]    = b
                        net_state["bal_at"] = time.time()
                except Exception:
                    pass
                finally:
                    with net_lock:
                        net_state["bal_inflight"] = False
            threading.Thread(target=_worker, daemon=True).start()

        def _reload_dashboard() -> None:
            """Full reset: restore the terminal and re-exec ourselves.
            Triggered by the `r` key and `/reload` — useful when the
            "⚠ network" banner is stuck (cached `rts_err`, stale
            in-memory state) and the only previous fix was Ctrl+C +
            restart.  Equivalent to that, just one keystroke."""
            try:
                if isatty and old_term is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            except Exception:
                pass
            try:
                sys.stdout.write(SHOW + WRAP_ON + ALT_OFF)
                sys.stdout.flush()
            except Exception:
                pass
            os.execv(sys.executable, [sys.executable, *sys.argv])

        while True:
            now = time.time()
            # Kick off a non-blocking refresh; render uses the last-known
            # result so we never freeze the dashboard on a slow socket.
            _async_fetch_rts()
            with net_lock:
                rts = list(net_state["rts"])
                rts_age = now - net_state["rts_at"] if net_state["rts_at"] else None
                rts_err = net_state["rts_err"]
            sh_map = _shorthand_map(rts)
            inv    = {ep: sh for sh, ep in sh_map.items()}
            jobs = store.list_jobs()

            # ── tail events.jsonl ─────────────────────────────────────────
            try:
                if _EVENTS_LOG.exists():
                    with _EVENTS_LOG.open() as f:
                        f.seek(events_pos); chunk = f.read(); events_pos = f.tell()
                    for ln in chunk.splitlines():
                        if not ln.strip(): continue
                        try:
                            ev = json.loads(ln)
                        except Exception:
                            continue
                        events_buf.append(ev)
            except Exception:
                pass
            active_eps = {r["endpoint"] for r in rts}
            open_jobs = [j for j in jobs if j.get("status") in ("queued", "running")
                                          and j.get("endpoint") in active_eps]

            # Balance: kick a background fetch at most once a minute, then
            # read whatever the latest successful result is.
            if now - last_balance_ts > 60:
                _async_fetch_balance()
                last_balance_ts = now
            with net_lock:
                last_balance = net_state["bal"]
            observed = _observed_rate_per_hour()

            # Compute burn
            total_rate = 0.0
            total_cost = 0.0
            rt_entries: list[dict] = []
            def _flatten_code_inline(code: str) -> str:
                parts = code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
                if not parts:
                    return ""
                return parts[0] + "".join(
                    "↵" + p.lstrip() for p in parts[1:] if p.strip()
                )
            for r in sorted(rts, key=lambda r:
                            (_runtime_meta(r["endpoint"]) or {}).get("assigned_at", 0)):
                ep = r["endpoint"]; sh = inv.get(ep, "?")
                accel = _normalized_accel(r)
                rate, hrs, cost = _runtime_cost(r)
                rate_s = f"{rate:.2f}/h" if rate is not None else "?"
                hrs_s  = f"{hrs:.2f}h" if hrs is not None else "?"
                cost_s = f"{cost:.2f}u" if cost is not None else "?"
                if rate: total_rate += rate
                if cost: total_cost += cost
                meta = _runtime_meta(ep) or {}
                desc = _trunc(meta.get("desc"), 20) or "—"
                tmo  = meta.get("idle_timeout_min")
                tmo_s = "—" if tmo is None else ("off" if tmo == 0 else f"{tmo}m")
                rt_entries.append({
                    "sh": sh, "ep": ep, "accel": accel,
                    "hrs_s": hrs_s, "rate_s": rate_s, "cost_s": cost_s,
                    "tmo_s": tmo_s, "desc": desc,
                })

            # Adaptive age formatter — keeps the column compact while
            # surfacing the right precision for the magnitude.
            #   • <60s   → "Ns"
            #   • <60m   → "MmSS" if M<10 else "Mm"     (e.g. "5m12", "47m")
            #   • <24h   → "HhMM" if H<10 else "Hh"     (e.g. "9h05", "23h")
            #   • ≥24h   → "DdHH" if D<10 else "Dd"     (e.g. "2d09", "30d")
            def _fmt_age_compact(seconds: float) -> str:
                if seconds < 0:     return "—"
                if seconds < 60:    return f"{int(seconds)}s"
                if seconds < 3600:
                    m, s = divmod(int(seconds), 60)
                    return f"{m}m{s:02d}" if m < 10 else f"{m}m"
                if seconds < 86400:
                    h, rem = divmod(int(seconds), 3600)
                    m = rem // 60
                    return f"{h}h{m:02d}" if h < 10 else f"{h}h"
                d, rem = divmod(int(seconds), 86400)
                h = rem // 3600
                return f"{d}d{h:02d}" if d < 10 else f"{d}d"

            # Open-job entries — store full desc/code untruncated so the
            # renderer can size them differently for overview vs focused view.
            # Code is flattened to one line: newlines become a single "↵"
            # sentinel char (rendered as a faded return-arrow at draw time)
            # and continuation-line indent is stripped so it reads clean.
            job_entries: list[dict] = []
            for j in sorted(open_jobs, key=lambda r: r.get("started", 0)):
                jid = j["job_id"]; ep = j.get("endpoint", "")
                sh  = inv.get(ep, "?")
                started = j.get("started") or now
                age_s = _fmt_age_compact(now - started)
                code_full = _flatten_code_inline(j.get("code") or "")
                desc_full = j.get("desc") or ""
                job_entries.append({
                    "sh": sh, "jid": jid, "status": j["status"],
                    "age_s": age_s, "desc_full": desc_full, "code_full": code_full,
                    "conn": _job_connection_status(j),
                })

            # All-job entries — pre-built once per outer-loop tick so the
            # `/jobs all` view doesn't pay disk + JSON cost on every
            # keystroke during a scroll burst.  This was the dominant
            # source of arrow-key lag on big histories.
            all_job_entries: list[dict] = []
            for j in sorted(jobs, key=lambda r: r.get("started", 0) or 0,
                            reverse=True)[:200]:
                ep_ = j.get("endpoint", "")
                sh_ = inv.get(ep_, "—")
                started_ = j.get("started") or 0
                age_s = (_fmt_age_compact(now - started_) if started_ else "—")
                all_job_entries.append({
                    "sh": sh_, "jid": j["job_id"],
                    "status": j.get("status", "?"),
                    "age_s": age_s,
                    "desc_full": j.get("desc") or "",
                    "code_full": _flatten_code_inline(j.get("code") or ""),
                    "conn": _job_connection_status(j),
                })

            # Recent output: tail each open job's events.jsonl since last render,
            # append new text lines into a global ring buffer.  First time we see
            # a job we seek near end-of-file so we don't dump megabytes of history.
            current_open = {j["job_id"] for j in open_jobs}
            for jid in list(last_pos.keys()):
                if jid not in current_open:
                    last_pos.pop(jid, None)        # clean up finished jobs
            for j in open_jobs:
                jid = j["job_id"]
                ep  = j.get("endpoint", "")
                sh  = inv.get(ep, "?")
                col = _color_for_letter(sh)
                path = store.events_path(jid)
                if not path.exists():
                    continue
                if jid not in last_pos:
                    # Seed near end-of-file so we don't replay all history.
                    try:
                        size = path.stat().st_size
                        last_pos[jid] = max(0, size - 4096)
                    except Exception:
                        last_pos[jid] = 0
                try:
                    with path.open() as f:
                        f.seek(last_pos[jid])
                        chunk = f.read()
                        last_pos[jid] = f.tell()
                except Exception:
                    continue
                for ev_line in chunk.splitlines():
                    if not ev_line.strip(): continue
                    try:
                        ev = json.loads(ev_line)
                    except Exception:
                        continue
                    text   = ev.get("text", "")
                    is_err = ev.get("type") in ("stderr", "error")
                    text_color = 91 if is_err else None
                    # tqdm progress bars use `\r` to overwrite the same row
                    # in-place (no `\n`).  Naively splitting on \r turns every
                    # update into a new line and the buffer fills with stale
                    # snapshots.  Walk the text instead: \n finalizes a line,
                    # \r overwrites the in-progress one, trailing partial is
                    # treated as in-progress so the next event can replace it.
                    # Entries are 6-tuples: (sh, jid, text, pfx_color, txt_color, in_progress).
                    def _commit(line: str, in_progress: bool) -> None:
                        # Strip ANSI escape sequences before the emptiness
                        # check.  tqdm/rich/etc. emit lines like "\x1b[K"
                        # (clear-to-end-of-line) between bar updates;
                        # `line.strip()` keeps them (they're not whitespace),
                        # so without this filter we'd commit blank-rendering
                        # entries that show up as empty rows in --live's
                        # follow view.
                        visible = _ANSI_CSI.sub("", line).strip()
                        if not visible: return
                        truncated = _ANSI_CSI.sub("", line)[:88]
                        new_entry = (sh, jid, truncated, col, text_color, in_progress)
                        # Find the most-recent in-progress entry for this jid.
                        for i in range(len(recent_output) - 1, -1, -1):
                            e = recent_output[i]
                            if len(e) >= 6 and e[1] == jid and e[5]:
                                if in_progress:
                                    # New tqdm tick — overwrite in place.
                                    recent_output[i] = new_entry
                                    return
                                # Finalizing past it — drop the stale bar
                                # snapshot, then append the new finalized line.
                                del recent_output[i]
                                break
                        recent_output.append(new_entry)
                    text = text.replace("\r\n", "\n")
                    line_buf = ""
                    for ch in text:
                        if ch == "\n":
                            _commit(line_buf, in_progress=False)
                            line_buf = ""
                        elif ch == "\r":
                            _commit(line_buf, in_progress=True)
                            line_buf = ""
                        else:
                            line_buf += ch
                    if line_buf:
                        _commit(line_buf, in_progress=True)

            # ── reusable renderers ─────────────────────────────────────────
            def render_runtimes(with_endpoint: bool = True,
                                with_timeout: bool = True,
                                with_cost: bool = True,
                                show_all: bool = False) -> list[str]:
                # Build entries: live `rt_entries` (active) plus, when show_all
                # is True, released-but-still-tracked endpoints from
                # runtimes.json.  Released entries have no live API row, so
                # we synthesize a partial dict.
                entries = list(rt_entries)
                if show_all:
                    seen = {e["ep"] for e in entries}
                    try:
                        metas = _runtime_meta_all() or {}
                    except Exception:
                        metas = {}
                    for ep, m in metas.items():
                        if ep in seen or not m.get("released_at"):
                            continue
                        entries.append({
                            "sh":     "—",
                            "ep":     ep,
                            "accel":  (m.get("accelerator") or "?")[:5],
                            "hrs_s":  "released",
                            "rate_s": "—",
                            "cost_s": "—",
                            "tmo_s":  "—",
                            "desc":   _trunc(m.get("desc"), 25) or
                                      f"released_by={m.get('released_by','?')}",
                        })
                if not entries:
                    return ["  \033[2mNo active instances.\033[0m\n"]
                header = f"  {'SH':<2}"
                if with_endpoint:
                    header += f"  {'ENDPOINT':<40}"
                header += f"  {'ACCEL':<5}  {'UPTIME':<7}"
                if with_cost:
                    header += f"  {'RATE':<8}  {'COST':<7}"
                if with_timeout:
                    header += f"  {'TIMEOUT':<7}"
                header += "  DESC"
                title = (f"  \033[1m{'All instances' if show_all else 'Active instances'} "
                         f"({len(entries)}):\033[0m\n")
                out = [title, header + "\n"]
                for e in entries:
                    sh = e["sh"]
                    line = f"  \033[{_color_for_letter(sh)}m{sh:<2}\033[0m"
                    if with_endpoint:
                        line += f"  {e['ep'][:40]:<40}"
                    line += f"  {e['accel']:<5}  {e['hrs_s']:<7}"
                    if with_cost:
                        line += f"  {e['rate_s']:<8}  {e['cost_s']:<7}"
                    if with_timeout:
                        line += f"  {e['tmo_s']:<7}"
                    line += f"  {e['desc']}"
                    out.append(line + "\n")
                return out

            def render_jobs(with_code: bool = True,
                            desc_w: int = 40, code_w: int = 80,
                            show_all: bool = False) -> list[str]:
                # Pick the precomputed entry list (built once per outer-loop
                # data refresh).  Avoids hitting disk / JSON-parsing the job
                # index on every keystroke.
                entries = all_job_entries if show_all else job_entries
                if not entries:
                    return ["  \033[2mNo open jobs.\033[0m\n"]
                header = (f"     {'SH':<2}  {'JOB_ID':<8}  {'STATUS':<10}  "
                          f"{'AGE':<6}  {'DESC':<{desc_w}}")
                if with_code:
                    header += "  CODE"
                title = (f"  \033[1m{'All jobs' if show_all else 'Open jobs'} "
                         f"({len(entries)}):\033[0m\n")
                out = [title, header + "\n"]
                for e in entries:
                    sh = e["sh"]
                    desc = _trunc(e["desc_full"], desc_w) or "—"
                    conn = e.get("conn", "n/a")
                    dot_color = {"connected": 32, "stale": 33,
                                 "disconnected": 31, "queued": 36}.get(conn, 90)
                    dot = f"\033[{dot_color}m●\033[0m"
                    # If conn says queued (kernel hasn't picked our request
                    # up yet even though watcher set status=running), show
                    # the user "queued" instead of misleading "running".
                    status_disp = "queued" if conn == "queued" else e["status"]
                    line = (f"  {dot}  \033[{_color_for_letter(sh)}m{sh:<2}\033[0m  "
                            f"{e['jid']}  {status_disp:<10}  {e['age_s']:<6}  "
                            f"{desc:<{desc_w}}")
                    if with_code:
                        # Truncate by char count (↵ is 1 char), then fade the
                        # return-arrow newline markers so they read as soft
                        # separators instead of part of the code.
                        code_chunk = e["code_full"][:code_w]
                        code_styled = code_chunk.replace("↵", "\033[2m↵\033[0m")
                        line += f"  {code_styled}"
                    out.append(line + "\n")
                return out

            def render_follow(limit: int | None = RECENT_MAX) -> list[str]:
                src = list(recent_output)
                if follow_filter:
                    src = [e for e in src if (e[0] or "").lower() == follow_filter.lower()]
                tail = src if limit is None else src[-limit:]
                if not tail:
                    msg = (f"  \033[2mNo recent output for runtime "
                           f"'{follow_filter}'.\033[0m\n" if follow_filter
                           else "  \033[2mNo recent output.\033[0m\n")
                    return [msg]
                hdr = (f"  \033[1mRecent output (last {len(tail)}, filter={follow_filter!r}):\033[0m\n"
                       if follow_filter
                       else f"  \033[1mRecent output (last {len(tail)}):\033[0m\n")
                out = [hdr]
                for entry in tail:
                    # Tolerate both 5-tuple (legacy) and 6-tuple (current with
                    # in_progress flag).  We don't render in_progress specially.
                    sh_, jid_, text_, prefix_color, text_color = entry[:5]
                    pfx  = f"\033[{prefix_color}m[{sh_:<2} {jid_}]\033[0m"
                    body = f"\033[{text_color}m{text_}\033[0m" if text_color else text_
                    out.append(f"  {pfx}  {body}\n")
                return out

            def render_events(limit: int | None) -> list[str]:
                tail = list(events_buf) if limit is None else list(events_buf)[-limit:]
                if not tail:
                    return ["  \033[2mNo events yet.\033[0m\n"]
                out = [f"  \033[1mEvents (last {len(tail)}):\033[0m\n"]
                for ev in tail:
                    t   = ev.get("type", "?")
                    ts  = time.strftime("%H:%M:%S", time.localtime(ev.get("ts", 0)))
                    jid = ev.get("jid", "")
                    ep  = ev.get("endpoint", "")
                    sh  = inv.get(ep, "?")
                    color = (
                        91 if t in ("job_error",) else
                        91 if t == "runtime_released" and ev.get("reason") == "preempted" else
                        33 if t in ("job_cancelled", "job_timeout", "runtime_released") else
                        32 if t in ("job_done", "runtime_assigned", "job_started") else 36
                    )
                    summary_bits = []
                    if jid: summary_bits.append(jid)
                    if sh != "?" and ep: summary_bits.append(f"[{sh}]")
                    if ev.get("reason"): summary_bits.append(f"reason={ev['reason']}")
                    if ev.get("elapsed_s") is not None: summary_bits.append(f"{ev['elapsed_s']}s")
                    summary = "  ".join(summary_bits)
                    out.append(f"  \033[{color}m{ts}  {t:<18}\033[0m  {summary}\n")
                return out

            def render_cost() -> list[str]:
                # Reuse the full runtime table — cost view is its own focused
                # screen, so the endpoint hash is genuinely useful here.
                if not rt_entries:
                    return ["  \033[2mNo active instances — no burn.\033[0m\n"]
                out = list(render_runtimes(with_endpoint=True))
                # Replace the leading "Active runtimes" header with a cost-styled one.
                out[0] = f"  \033[1mPer-instance cost ({len(rt_entries)}):\033[0m\n"
                out.append(f"\n  TOTAL  burn={total_rate:.2f} u/h    spent={total_cost:.2f} u\n")
                return out

            # ── render ─────────────────────────────────────────────────────
            import re
            _ANSI = re.compile(r"\x1b\[[0-9;]*m")
            def _vis_len(s: str) -> int:
                return len(_ANSI.sub("", s))
            def _clip_ansi(s: str, width: int) -> str:
                if width <= 0:
                    return ""
                out: list[str] = []
                vis = 0
                i = 0
                saw_ansi = False
                while i < len(s) and vis < width:
                    if s[i] == "\x1b" and i + 1 < len(s) and s[i + 1] == "[":
                        j = i + 2
                        while j < len(s) and not ("@" <= s[j] <= "~"):
                            j += 1
                        if j < len(s):
                            out.append(s[i:j + 1])
                            saw_ansi = True
                            i = j + 1
                            continue
                    ch = s[i]
                    if ch in "\r\n":
                        break
                    out.append(ch)
                    vis += 1
                    i += 1
                clipped = "".join(out)
                if saw_ansi:
                    clipped += "\033[0m"
                return clipped
            def _side_by_side(left: list[str], right: list[str], gap: str = "    ") -> list[str]:
                rows = max(len(left), len(right), 1)
                left  = left  + [""] * (rows - len(left))
                right = right + [""] * (rows - len(right))
                target = max((_vis_len(l) for l in left), default=0)
                return [
                    f"{l}{' ' * (target - _vis_len(l))}{gap}{r}"
                    for l, r in zip(left, right)
                ]

            def _pad_section(block: list[str], target: int) -> list[str]:
                """Strip newlines and clip / pad to exactly `target` lines."""
                stripped = [ln.rstrip("\n") for ln in block]
                if len(stripped) >= target:
                    return stripped[:target]
                return stripped + [""] * (target - len(stripped))

            # Footer is a constant string — built once per render.
            footer_line = (
                "  \033[2m"
                "[\033[0m\033[1mo\033[0m\033[2m]verview  "
                "[\033[0m\033[1mi\033[0m\033[2m]nstances  "
                "[\033[0m\033[1mj\033[0m\033[2m]obs  "
                "[\033[0m\033[1mc\033[0m\033[2m]ost  "
                "[\033[0m\033[1mf\033[0m\033[2m]ollow  "
                "[\033[0m\033[1me\033[0m\033[2m]vents  "
                "[\033[0m\033[1mr\033[0m\033[2m]eload  "
                "[\033[0m\033[1m/\033[0m\033[2m]cmd  "
                "[\033[0m\033[1mq\033[0m\033[2m]uit\033[0m"
            )

            def _palette_completions(buf_str: str) -> list[tuple[str, str]]:
                """Context-sensitive autocomplete entries for the palette.

                If the user is still typing the command name, returns matching
                command name entries (each value already includes the leading
                ``/``).  Once the cursor is past a space, returns argument
                values appropriate to the command:

                * ``/cancel`` / ``/status``         → recent job ids
                * ``/release`` / ``/timeout`` / ``/run`` → runtime shorthands
                * ``/desc``                         → runtimes + job ids
                * ``/assign``                       → accelerator names
                * ``/env``                          → sub-op (list/set/rm/show),
                                                       then key names from .env

                Returns up to 12 entries.  Entries are ``(value, description)``.
                """
                parts = buf_str.split(" ")
                if len(parts) == 1:
                    pref = parts[0].lower()
                    return [(f"/{n}", f"{p}  — {d}" if p else d)
                            for (n, p, d) in _PALETTE_CMDS
                            if n.startswith(pref)]
                cmd = parts[0].lower()
                idx = len(parts) - 1
                arg_pref = parts[-1]
                arg_pref_l = arg_pref.lower()

                def _rt_sugg(p: str) -> list[tuple[str, str]]:
                    out: list[tuple[str, str]] = []
                    sh_to_ep = {sh: ep for ep, sh in inv.items()}
                    for sh in sorted(sh_to_ep):
                        ep = sh_to_ep[sh]
                        accel = "?"
                        for r in rts:
                            if r["endpoint"] == ep:
                                accel = _normalized_accel(r) or "?"
                                break
                        if sh.lower().startswith(p):
                            out.append((sh, f"{accel}  {ep[:30]}"))
                    seen = set()
                    for r in rts:
                        a = _normalized_accel(r) or ""
                        if a and a not in seen:
                            seen.add(a)
                            if a.lower().startswith(p):
                                out.append((a, f"all {a} runtimes"))
                    if "all".startswith(p) and rts:
                        out.append(("all", f"all {len(rts)} active runtimes"))
                    return out

                def _jid_sugg(p: str) -> list[tuple[str, str]]:
                    try:
                        jl = store.list_jobs()
                    except Exception:
                        jl = []
                    jl = sorted(jl, key=lambda r: r.get("started", 0), reverse=True)[:50]
                    out: list[tuple[str, str]] = []
                    for j in jl:
                        if not j["job_id"].startswith(p): continue
                        # Apply the same display override the dashboard uses:
                        # the store flips status="running" the moment the
                        # watcher attaches even if the kernel is still busy
                        # with a prior cell.  Surface that as "queued".
                        conn = _job_connection_status(j)
                        st = "queued" if conn == "queued" else j.get("status", "?")
                        out.append((j["job_id"],
                                    f"{st:<9}  {(j.get('desc') or '')[:30]}"))
                        if len(out) >= 12: break
                    return out

                if cmd in ("cancel", "status", "reattach"):
                    return _jid_sugg(arg_pref) if idx == 1 else []
                if cmd in ("release", "unassign"):
                    return _rt_sugg(arg_pref_l) if idx == 1 else []
                if cmd in ("timeout", "set-timeout"):
                    return _rt_sugg(arg_pref_l) if idx == 1 else []
                if cmd == "desc":
                    if idx == 1:
                        return _rt_sugg(arg_pref_l) + _jid_sugg(arg_pref)
                    return []
                if cmd in ("run", "submit"):
                    return _rt_sugg(arg_pref_l) if idx == 1 else []
                if cmd == "assign":
                    if idx == 1:
                        accels = ["CPU", "T4", "L4", "A100", "H100", "G4"]
                        return [(a, "") for a in accels
                                if a.lower().startswith(arg_pref_l)]
                    return []
                if cmd == "env":
                    if idx == 1:
                        ops = [("list", "show all keys"),
                               ("set",  "set NAME=VALUE"),
                               ("rm",   "remove a key"),
                               ("show", "print a value")]
                        return [(o, d) for (o, d) in ops if o.startswith(arg_pref_l)]
                    if idx == 2:
                        sub = parts[1].lower()
                        try:
                            env = _load_env()
                        except Exception:
                            env = {}
                        if sub in ("rm", "show"):
                            return [(k, f"{len(env[k])} chars") for k in sorted(env)
                                    if k.startswith(arg_pref)][:12]
                        if sub == "set":
                            return [(f"{k}=", f"current: {len(env[k])} chars")
                                    for k in sorted(env)
                                    if k.startswith(arg_pref)][:12]
                    return []
                return []

            def do_render() -> None:
                """Repaint the dashboard.  Layout shape (rows top-to-bottom):

                    1   title
                    1   balance line (if room)
                    B   dashboard body
                    1   footer

                When the `/` panel is active, it overlays rows directly under
                the title without changing the base dashboard layout.
                """
                import shutil
                nonlocal panel_scroll, last_render_at, palette_sel
                last_render_at = time.time()
                try:
                    term_size = shutil.get_terminal_size()
                    term_rows = term_size.lines
                    term_cols = term_size.columns
                except Exception:
                    term_rows = 30
                    term_cols = 80

                top_lines: list[str] = []

                # Offline / stale-data indicator: if the last successful
                # `list_runtimes` was > 5 s ago (or never), surface that in
                # the title row so the user knows what they're looking at.
                stale_tag = ""
                if rts_err and (rts_age is None or rts_age > 5):
                    stale_tag = (f"  \033[33m⚠ network ({rts_err}) — "
                                 f"showing data from "
                                 f"{int(rts_age) if rts_age is not None else '?'}"
                                 f"s ago\033[0m")
                top_lines.append(
                    f"\033[1;36m●  colab live — {view}  "
                    f"({time.strftime('%H:%M:%S')})\033[0m{stale_tag}"
                )
                panel_active = bool(cmd_input is not None or
                                    (last_cmd_lines and time.time() < last_cmd_until))
                bal = f"{last_balance:.2f}" if last_balance is not None else "?"
                obs = f"{observed:.2f}" if observed is not None else "—"
                balance_line = (
                    f"  Balance: {bal} units    Burn (est): {total_rate:.2f} u/h    "
                    f"Observed: {obs} u/h    Spent so far: {total_cost:.2f} u"
                )
                if term_rows >= 4:
                    top_lines.append(balance_line)
                top_capacity = max(0, term_rows - 2)
                top_lines = top_lines[:top_capacity]

                panel_lines: list[str] = []
                if cmd_input is not None:
                    parts = cmd_input.split(" ")
                    if len(parts) == 1:
                        hint = "Tab complete cmd · Enter run · Esc cancel · /help"
                    else:
                        hint = "Tab complete arg · Enter run · Esc cancel"
                    # Compute inline ghost preview: what Tab would fill in
                    # right now, rendered very dim before the cursor (Fish /
                    # Google-search style).  Only the unwritten suffix is
                    # shown — the part the user has already typed isn't
                    # duplicated.  Skipped if the would-be value doesn't
                    # extend the current trailing token.  We cache the
                    # `_palette_completions` result here and reuse it for
                    # the suggestion list below — calling it twice doubled
                    # autocomplete-scroll latency (Up/Down felt laggy on
                    # `/status ` over a long job history).
                    ghost = ""
                    cached_sugg: list[tuple[str, str]] | None = None
                    if (cmd_input != "" or palette_show_all) and palette_autocomplete:
                        cached_sugg = _palette_completions(cmd_input)
                        if cached_sugg:
                            sel_p = max(0, min(palette_sel, len(cached_sugg) - 1))
                            sel_value = cached_sugg[sel_p][0]
                            last_token = parts[-1]
                            if len(parts) == 1:
                                full = sel_value.lstrip("/")
                                if full.startswith(last_token):
                                    ghost = full[len(last_token):]
                            else:
                                if sel_value.startswith(last_token):
                                    ghost = sel_value[len(last_token):]
                    ghost_seg = f"\033[2m{ghost}\033[0m" if ghost else ""
                    panel_lines.append(
                        f"  \033[1;36m> /\033[0m{cmd_input}{ghost_seg}\033[7m \033[0m  "
                        f"\033[2m({hint})\033[0m"
                    )
                    if (cmd_input != "" or palette_show_all) and palette_autocomplete:
                        # Pinned signature line for the matched command
                        # (stays visible after the user types a space, so
                        # `/cancel <jid> — …` keeps reminding the format).
                        # Bold-only — no color — so it reads as a label,
                        # not a competing accent.
                        parts_dbg = cmd_input.split(" ")
                        sig_line: str | None = None
                        if len(parts_dbg) > 1:
                            cmd_typed = parts_dbg[0].lower()
                            for (nm, sig, desc) in _PALETTE_CMDS:
                                if nm == cmd_typed:
                                    label = f"/{nm}" + (f" {sig}" if sig else "")
                                    sig_line = (f"    \033[1m{label:<24}\033[0m"
                                                f"  \033[2m{desc}\033[0m")
                                    break
                        if sig_line is not None:
                            panel_lines.append(sig_line)
                        sugg = cached_sugg if cached_sugg is not None else _palette_completions(cmd_input)
                        sugg_cap = 11 if sig_line is not None else 12
                        # Clamp the selection cursor (set elsewhere by the
                        # Up/Down arrow handlers in palette mode) so it
                        # never points past the visible window.
                        palette_sel = max(0, min(palette_sel, max(0, len(sugg[:sugg_cap]) - 1)))
                        for i, (value, desc) in enumerate(sugg[:sugg_cap]):
                            line = f"{value:<24}  {desc}" if desc else value
                            if i == palette_sel:
                                # Selected row: leading arrow + bold (no
                                # reverse-video — the white-block was too
                                # loud against the dashed body).
                                panel_lines.append(f"  \033[1m▸ {line}\033[0m")
                            else:
                                panel_lines.append(f"    \033[2m{line}\033[0m")
                elif last_cmd_lines and time.time() < last_cmd_until:
                    # Slice last_cmd_lines by panel_scroll so Up/Down arrows
                    # can pan a long /help (or any other multi-line output).
                    overlay_capacity = max(0, term_rows - 4)
                    max_panel_off = max(0, len(last_cmd_lines) - overlay_capacity)
                    panel_scroll = max(0, min(panel_scroll, max_panel_off))
                    end = panel_scroll + overlay_capacity
                    visible = last_cmd_lines[panel_scroll:end]
                    panel_lines = [f"  \033[2m{ln}\033[0m" for ln in visible]
                    if max_panel_off > 0:
                        n_above = panel_scroll
                        n_below = max(0, len(last_cmd_lines) - end)
                        panel_lines.append(
                            f"  \033[2m[\033[0m\033[36m↑ {n_above} above\033[0m"
                            f"\033[2m  ·  \033[0m\033[36m↓ {n_below} below\033[0m"
                            f"\033[2m]\033[0m"
                        )

                panel_separator = "  \033[2m----- end cmd; dashboard below -----\033[0m"
                body_height = max(0, term_rows - len(top_lines) - 2)

                content: list[str] = []
                if view == "overview":
                    overview_base_rows = 13
                    overview_extra_rows = max(0, body_height - overview_base_rows)
                    events_rows = 3
                    follow_rows = 3 + overview_extra_rows
                    content.append("")
                    rt = _pad_section(
                        render_runtimes(with_endpoint=False, with_timeout=False,
                                        with_cost=False), 4)
                    jb = _pad_section(
                        render_jobs(with_code=False, desc_w=20), 4)
                    for line in _side_by_side(rt, jb):
                        content.append(line)
                    content.append("")
                    content.extend(_pad_section(
                        render_events(max(2, events_rows - 1)), events_rows))
                    content.append("")
                    content.extend(_pad_section(
                        render_follow(limit=max(2, follow_rows - 1)), follow_rows))
                elif view in scrollable_views:
                    # Build the full content for the active view, then slice
                    # body items by `scroll_offsets[view]` while keeping the
                    # title/header rows fixed.  `header_n` matches each
                    # renderer's leading-line count.
                    if view == "runtimes":
                        full = render_runtimes(show_all=view_all)
                        header_n = 2
                    elif view == "jobs":
                        full = render_jobs(show_all=view_all)
                        header_n = 2
                    elif view == "cost":
                        full = render_cost()
                        header_n = 2 if len(full) >= 2 else len(full)
                    elif view == "follow":
                        full = render_follow(limit=None)
                        header_n = 1
                    else:    # events
                        full = render_events(None)
                        header_n = 1
                    # Leading blank row so the focus-view header sits one
                    # row below the balance line, matching the overview
                    # spacing.
                    content.append("")
                    full_lines = [ln.rstrip("\n") for ln in full]
                    if len(full_lines) <= header_n:
                        # Empty/no-data render — just show what we have.
                        content.extend(full_lines)
                        scroll_offsets[view] = 0
                    else:
                        items = full_lines[header_n:]
                        # Subtract 1 from the item budget for the leading
                        # blank row we just appended above.
                        raw_item_h = max(0, body_height - header_n - 1)
                        # If items overflow even the raw budget, we'll need
                        # an indicator row and have to give up one more
                        # row of items for it.  Compute final item_h FIRST
                        # so max_off matches what we'll actually display —
                        # otherwise the very last item is unreachable when
                        # the user scrolls to the bottom.
                        needs_indicator = len(items) > raw_item_h
                        item_h = (raw_item_h - 1) if (needs_indicator and raw_item_h > 1) else raw_item_h
                        max_off = max(0, len(items) - item_h)
                        if at_tail.get(view, False):
                            scroll_offsets[view] = max_off
                        off = max(0, min(scroll_offsets[view], max_off))
                        scroll_offsets[view] = off
                        # Re-arm tailing when the user manually scrolls back
                        # to the bottom of a tail-style view.
                        if view in ("follow", "events") and off == max_off:
                            at_tail[view] = True
                        n_above = off
                        n_below = max(0, len(items) - off - item_h)
                        content.extend(full_lines[:header_n])
                        content.extend(items[off:off + item_h])
                        if needs_indicator:
                            content.append(
                                f"  \033[2m[\033[0m\033[36m↑ {n_above} above\033[0m"
                                f"\033[2m  ·  \033[0m\033[36m↓ {n_below} below\033[0m"
                                f"\033[2m]\033[0m"
                            )

                if len(content) >= body_height:
                    content = content[:body_height]
                if len(content) < body_height:
                    content = content + [""] * (body_height - len(content))

                visible_lines: list[str] = []
                visible_lines.extend(_clip_ansi(line, term_cols) for line in top_lines)
                visible_lines.extend(_clip_ansi(line, term_cols) for line in content)
                visible_lines.append("")
                visible_lines.append(_clip_ansi(footer_line, term_cols))

                if panel_active and len(visible_lines) > 1:
                    overlay_rows = max(0, len(visible_lines) - 2)
                    overlay_lines = panel_lines[:overlay_rows]
                    if overlay_lines:
                        extra = overlay_rows - len(overlay_lines)
                        if extra > 0:
                            overlay_lines.append(panel_separator)
                            extra -= 1
                        overlay_lines = overlay_lines[:overlay_rows]
                        for i, line in enumerate(overlay_lines, start=1):
                            if i < len(visible_lines) - 1:
                                visible_lines[i] = _clip_ansi(line, term_cols)

                # Position each line at an absolute row + `\033[K` to clear
                # any residual chars to end-of-line.  This is more robust than
                # `"\n".join(...)` because it avoids any chance of a long line
                # soft-wrapping (or a stray embedded newline) shifting the
                # footer's row.  The footer always lands at row `term_rows`
                # and the row above it is always blank.
                buf = [HOME, CLEAR_BELOW]
                # Clamp visible_lines to exactly term_rows so the footer's
                # absolute row is deterministic.  Pre-clip width too.
                if len(visible_lines) > term_rows:
                    # Keep the title block, the empty separator, and the
                    # footer; trim from the body.  Footer = last entry.
                    body_keep = max(0, term_rows - len(top_lines) - 2)
                    keep = (visible_lines[:len(top_lines)]
                            + visible_lines[len(top_lines):len(top_lines) + body_keep]
                            + ["", visible_lines[-1]])
                    visible_lines = keep[:term_rows]
                elif len(visible_lines) < term_rows:
                    # Pad before the empty-separator/footer pair so they stay last.
                    pad = term_rows - len(visible_lines)
                    visible_lines = (visible_lines[:-2] + [""] * pad +
                                     visible_lines[-2:])
                for row, line in enumerate(visible_lines, start=1):
                    buf.append(f"\033[{row};1H")
                    buf.append(_clip_ansi(line, term_cols))
                    buf.append("\033[K")
                sys.stdout.write("".join(buf)); sys.stdout.flush()

            def run_palette_cmd(line: str) -> list[str]:
                """Dispatch one /-prefix-stripped command line.  Returns
                lines to display.  Operates on the closure's `store`, `sess`,
                `rts`, `inv`.  May raise KeyboardInterrupt to quit.
                """
                nonlocal view, follow_filter, view_all, palette_autocomplete
                if not line: return []
                head, _sep, tail = line.partition(" ")
                name = head.lstrip("/").lower()
                rest = tail.strip()
                if name in ("help", "h", "?"):
                    out = ["commands:"]
                    for nm, sig, desc in _PALETTE_CMDS:
                        label = f"/{nm}" + (f" {sig}" if sig else "")
                        out.append(f"  {label:<46}{desc}")
                    out.append("aliases: /q /exit → /quit · /unassign → /release · "
                               "/submit → /run · /url → /notebook-url · /set-timeout → /timeout")
                    out.append("<runtime> accepts a letter (a/b/…), accelerator (A100), "
                               "endpoint prefix, or 'all'.")
                    return out
                if name in ("q", "quit", "exit"):
                    raise KeyboardInterrupt
                if name == "cancel":
                    if not rest: return ["usage: /cancel <jid>"]
                    rec = store.get_job(rest)
                    if not rec: return [f"unknown job: {rest}"]
                    if rec["status"] not in ("queued", "running"):
                        return [f"job already {rec['status']}"]
                    # Either status="queued" OR running-but-not-actually-
                    # executing (watcher attached, kernel still on a prior
                    # cell).  Either way, no interrupt — that would kill
                    # whatever IS executing, not this job.
                    if rec["status"] == "queued" or _job_connection_status(rec) == "queued":
                        store.update_job(rest, status="cancelled", ended=time.time())
                        store.append_event(rest, {"type": "stderr",
                            "text": "\n[direct_kernel] cancelled before start (was queued)\n"})
                        return [f"cancelled {rest} (queued — no interrupt sent)"]
                    fresh = _refresh_proxy_for_endpoint(sess, rec["endpoint"])
                    if fresh is None:
                        store.update_job(rest, status="cancelled", ended=time.time())
                        return [f"cancelled {rest} (runtime gone)"]
                    jurl, ptok = fresh
                    store.update_job(rest, status="cancelled", ended=time.time())
                    store.append_event(rest, {"type": "stderr",
                        "text": "\n[direct_kernel] cancelled via /cancel\n"})
                    ok = _do_interrupt(sess, jurl, ptok, rec["kernel_id"])
                    return [f"cancelled {rest} ({'ok' if ok else 'http error'})"]
                if name in ("release", "unassign"):
                    if not rest: return ["usage: /release <runtime>"]
                    try:
                        eps = _resolve_endpoints(rest, rts)
                    except ValueError as e:
                        return [str(e)]
                    msgs = []
                    for ep in eps:
                        try:
                            unassign_runtime(sess, ep)
                            msgs.append(f"released {inv.get(ep, ep[:8])}  ({ep[:38]})")
                        except Exception as e:
                            msgs.append(f"release {ep[:8]} failed: {type(e).__name__}: {e}")
                    return msgs or ["no runtimes matched"]
                if name in ("timeout", "set-timeout"):
                    toks = rest.split()
                    if len(toks) != 2: return ["usage: /timeout <runtime> <min>"]
                    try:
                        mins = int(toks[1])
                    except ValueError:
                        return ["min must be an integer"]
                    try:
                        eps = _resolve_endpoints(toks[0], rts)
                    except ValueError as e:
                        return [str(e)]
                    msgs = []
                    for ep in eps:
                        _set_runtime_meta(ep, idle_timeout_min=mins)
                        try: _spawn_reaper(ep)
                        except Exception: pass
                        lab = "off" if mins == 0 else f"{mins}m"
                        msgs.append(f"{inv.get(ep, ep[:8])} idle-timeout → {lab}")
                    return msgs
                if name == "desc":
                    toks = rest.split(maxsplit=1)
                    if len(toks) < 2: return ["usage: /desc <rt|jid> <text>"]
                    target, text = toks[0], toks[1]
                    if store.get_job(target):
                        store.update_job(target, desc=text)
                        return [f"job {target} desc → {text[:60]}"]
                    try:
                        eps = _resolve_endpoints(target, rts)
                    except ValueError as e:
                        return [str(e)]
                    if len(eps) != 1:
                        return [f"{target!r} matches {len(eps)} runtimes (need exactly 1)"]
                    _set_runtime_meta(eps[0], desc=text)
                    return [f"runtime {inv.get(eps[0], eps[0][:8])} desc → {text[:60]}"]
                if name == "assign":
                    if not rest: return ["usage: /assign <accel> [desc]"]
                    parts = rest.split(maxsplit=1)
                    accel = parts[0].upper()
                    desc_arg = parts[1] if len(parts) > 1 else None
                    cmd = [sys.executable, str(Path(__file__).resolve()),
                           "--assign", "-a", accel]
                    if desc_arg:
                        cmd += ["-d", desc_arg]
                    try:
                        import subprocess
                        subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                    except Exception as e:
                        return [f"spawn failed: {type(e).__name__}: {e}"]
                    extra = f" -d {desc_arg!r}" if desc_arg else ""
                    return [f"spawned: --assign -a {accel}{extra}",
                            "(check the runtimes view in 30-60 s for the new entry)"]
                if name in ("run", "submit"):
                    parts = rest.split(maxsplit=1)
                    if len(parts) < 2:
                        return ["usage: /run <runtime> <code>"]
                    rt_q, code = parts[0], parts[1]
                    try:
                        eps = _resolve_endpoints(rt_q, rts)
                    except ValueError as e:
                        return [str(e)]
                    if len(eps) != 1:
                        return [f"{rt_q!r} matches {len(eps)} runtimes (need exactly 1)"]
                    cmd = [sys.executable, str(Path(__file__).resolve()),
                           "--no-stream", "-r", eps[0], "-c", code]
                    try:
                        import subprocess
                        subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                    except Exception as e:
                        return [f"spawn failed: {type(e).__name__}: {e}"]
                    code_preview = _flatten_code_inline(code)
                    return [f"spawned detached job on {inv.get(eps[0], eps[0][:8])}",
                            f"  code: {code_preview[:80]}{'…' if len(code_preview) > 80 else ''}"]
                if name in ("overview", "runtimes", "instances", "list",
                           "jobs", "cost", "follow", "events"):
                    # `/instances` and `/list` are user-facing aliases for
                    # `/runtimes`; the internal view name stays "runtimes"
                    # so scroll_offsets / at_tail keys / etc. don't fork.
                    view = "runtimes" if name in ("list", "instances") else name
                    if name == "follow":
                        follow_filter = (rest.strip() or None)
                    if name in ("jobs", "runtimes", "instances", "list"):
                        view_all = (rest.strip().lower() in ("all", "--all"))
                    else:
                        view_all = False
                    # Empty return → no panel output → no overlay covering
                    # the top rows of the view the user just navigated to.
                    # The view change itself is the visible feedback.
                    return []
                if name == "latest":
                    try:
                        jl = sorted(store.list_jobs(),
                                    key=lambda j: j.get("started", 0) or 0,
                                    reverse=True)
                    except Exception:
                        jl = []
                    running = [j for j in jl if j.get("status") == "running"]
                    target = running[0] if running else (jl[0] if jl else None)
                    if not target:
                        return ["no jobs found"]
                    sh = inv.get(target.get("endpoint", ""), None)
                    view = "follow"
                    follow_filter = sh
                    return [f"view: follow  (latest job {target['job_id']} on runtime {sh!r})"]
                if name == "reattach":
                    if not rest:
                        return ["usage: /reattach <jid>"]
                    jid = rest.strip()
                    rec = store.get_job(jid)
                    if rec is None:
                        return [f"unknown job_id: {jid!r}"]
                    if rec.get("status") in ("done", "error", "cancelled"):
                        return [f"job {jid} already terminal: {rec['status']}"]
                    cmd = [sys.executable, str(Path(__file__).resolve()),
                           "--reattach", jid, "--no-stream"]
                    try:
                        import subprocess
                        subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL,
                                         start_new_session=True)
                    except Exception as e:
                        return [f"spawn failed: {type(e).__name__}: {e}"]
                    return [f"reattach daemon spawned for job {jid}",
                            f"  events stream into {jid}.events.jsonl with `_reattached: true` flag"]
                if name == "status":
                    if not rest: return ["usage: /status <jid>"]
                    rec = store.get_job(rest.strip())
                    if not rec: return [f"unknown job: {rest}"]
                    fmt_t = lambda t: (time.strftime("%Y-%m-%d %H:%M:%S",
                                                     time.localtime(t)) if t else "—")
                    code1 = (_flatten_code_inline(rec.get("code") or "")[:120]
                             if rec.get("code") else "—")
                    conn = _job_connection_status(rec)
                    conn_color = {"connected": 32, "stale": 33,
                                  "disconnected": 31, "queued": 36}.get(conn, 90)
                    conn_line = f"  connection:  \033[{conn_color}m●\033[0m {conn}"
                    if conn == "disconnected":
                        conn_line += f"  \033[2m(use `/reattach {rec['job_id']}`)\033[0m"
                    status_disp = "queued" if conn == "queued" else rec.get("status", "?")
                    return [
                        f"job {rec['job_id']}: \033[1m{status_disp}\033[0m",
                        f"  endpoint:    {rec.get('endpoint','?')}",
                        f"  accelerator: {rec.get('accelerator','?')}",
                        f"  started:     {fmt_t(rec.get('started'))}",
                        f"  ended:       {fmt_t(rec.get('ended'))}",
                        f"  desc:        {rec.get('desc') or '—'}",
                        conn_line,
                        f"  code:        {code1}",
                    ]
                if name == "balance":
                    bal_now = _account_balance(sess)
                    if bal_now is None: return ["balance unavailable (API blip; try again)"]
                    return [f"balance: {bal_now:.2f} units"]
                if name == "event":
                    if not rest:
                        return ["usage: /event <type> [k=v ...]"]
                    parts_ev = rest.split()
                    ev_type = parts_ev[0]
                    fields: dict = {}
                    bad: list[str] = []
                    for kv in parts_ev[1:]:
                        k, _, v = kv.partition("=")
                        if not k or not _:
                            bad.append(kv); continue
                        fields[k] = v
                    _emit_event(ev_type, **fields)
                    msg = [f"emitted event type={ev_type!r}"]
                    if fields: msg.append(f"  fields: {fields}")
                    if bad:    msg.append(f"  ignored malformed: {bad}")
                    return msg
                if name == "reload":
                    _reload_dashboard()    # never returns (execv replaces)
                    return ["reloading…"]    # unreachable, but keeps mypy happy
                if name == "autocomplete":
                    arg = rest.strip().lower()
                    if arg in ("on", "1", "true", "yes", ""):
                        palette_autocomplete = True
                    elif arg in ("off", "0", "false", "no"):
                        palette_autocomplete = False
                    else:
                        return [f"usage: /autocomplete [on|off]  (currently: "
                                f"{'on' if palette_autocomplete else 'off'})"]
                    return [f"autocomplete: {'on' if palette_autocomplete else 'off'}"]
                if name == "refresh":
                    return ["forced refresh"]   # outer loop refetches immediately
                if name in ("notebook-url", "url"):
                    if not _NOTEBOOK_FILE.exists():
                        return ["no saved notebook id (run --test-cpu once to create)"]
                    try:
                        fid = _NOTEBOOK_FILE.read_text().strip()
                    except Exception as e:
                        return [f"read err: {e}"]
                    return [f"https://colab.research.google.com/drive/{fid}"]
                if name == "env":
                    sub = rest.split(maxsplit=1)
                    op  = sub[0].lower() if sub else "list"
                    arg = sub[1] if len(sub) > 1 else ""
                    if op == "list":
                        env = _load_env()
                        if not env: return ["(no .env keys)"]
                        keys = sorted(env.keys())
                        return [f"colab_bridge/.env  ({len(keys)} keys):"] + [
                            f"  {k}  ({len(env[k])} chars)" for k in keys
                        ]
                    if op == "set":
                        if "=" not in arg:
                            return ["usage: /env set NAME=VALUE",
                                    "  (palette doesn't accept multi-line values; use",
                                    "   `colab --env-set NAME --from-file PATH` for those)"]
                        k, _eq, v = arg.partition("=")
                        k = k.strip()
                        if not k: return ["bad name"]
                        env = _load_env(); env[k] = v
                        _save_env(env)
                        return [f"set {k}  ({len(v)} chars)"]
                    if op in ("rm", "remove", "del", "delete"):
                        k = arg.strip()
                        if not k: return ["usage: /env rm NAME"]
                        env = _load_env()
                        if k not in env: return [f"no such key: {k}"]
                        del env[k]; _save_env(env)
                        return [f"removed {k}"]
                    if op == "show":
                        k = arg.strip()
                        if not k: return ["usage: /env show NAME"]
                        env = _load_env()
                        if k not in env: return [f"no such key: {k}"]
                        v = env[k]
                        if "\n" in v:
                            head_lines = v.splitlines()[:8]
                            extra_n = len(v.splitlines()) - 8
                            out = [f"{k} = ({len(v)} chars, multi-line):"] + \
                                  ["  " + ln[:200] for ln in head_lines]
                            if extra_n > 0:
                                out.append(f"  …(+{extra_n} more lines)")
                            return out
                        if len(v) > 400:
                            return [f"{k} = {v[:400]}…  ({len(v)} chars total)"]
                        return [f"{k} = {v}"]
                    return [f"unknown env op: {op}  (list / set / rm / show)"]
                if name == "log":
                    n = 20
                    if rest:
                        try: n = int(rest)
                        except ValueError: return ["usage: /log [n]"]
                    if not _REAPER_LOG.exists():
                        return ["no reaper.log yet"]
                    lines_all = _REAPER_LOG.read_text().splitlines()
                    tail_lines = lines_all[-n:]
                    return [f"reaper.log (last {len(tail_lines)} of {len(lines_all)}):"] \
                        + ["  " + ln[:200] for ln in tail_lines]
                if name == "keepalive":
                    n = 20
                    if rest:
                        try: n = int(rest)
                        except ValueError: return ["usage: /keepalive [n]"]
                    if not _WS_COVERAGE_LOG.exists():
                        return ["no ws_coverage.log yet"]
                    lines_all = _WS_COVERAGE_LOG.read_text().splitlines()
                    tail_lines = lines_all[-n:]
                    return [f"ws_coverage.log (last {len(tail_lines)} of {len(lines_all)}):"] \
                        + ["  " + ln[:200] for ln in tail_lines]
                if name == "clear":
                    recent_output.clear()
                    events_buf.clear()
                    return ["cleared output + events buffers"]
                return [f"unknown command: /{name}  (try /help)"]

            do_render()

            # Wait up to 1 s for a keystroke, then loop back to refresh data.
            # On a view-switch key we redraw immediately with the cached
            # data — no waiting on `list_runtimes()` — so `o` / `r` / `j` /
            # `c` / `f` / `e` (plus ESC / SPACE for overview) feel snappy.
            deadline = time.time() + 1.0
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                if not isatty:
                    time.sleep(remaining); break
                # If `_read_key` left bytes buffered from a prior call (e.g.
                # the second `\x1b[B` of a back-to-back wheel-down pair),
                # process them immediately — don't wait on stdin for new
                # input.  Without this, queued events stay in the buffer
                # until the user happens to press another key, which is
                # the lag-builds-up symptom.
                if not _pending_buf:
                    try:
                        rsel, _, _ = select.select([sys.stdin], [], [], remaining)
                    except Exception:
                        rsel = []
                    if not rsel:
                        break
                key = _read_key(0.0)
                if key is None:
                    continue
                if key == "\x03" or key == "q":
                    raise KeyboardInterrupt

                if key == "/":
                    # Enter command-palette input mode.  Reads keystrokes
                    # one at a time until Enter (run) / Esc (cancel) / Ctrl+C.
                    cmd_input = ""
                    palette_show_all = False    # set True when user Tab's on empty buffer
                    palette_sel = 0             # arrow-keys move this; Tab fills sugg[palette_sel]
                    do_render()
                    while cmd_input is not None:
                        ch = _read_key(1.0)
                        if ch is None:
                            continue
                        if ch == "\x03":
                            raise KeyboardInterrupt
                        if ch == "\x1b":           # Esc cancels input
                            cmd_input = None
                            break
                        if ch in ("__UP__", "__WHEEL_UP__"):
                            palette_sel = max(0, palette_sel - 1)
                            # Coalesce held arrows: skip the redraw if more
                            # keys are already queued — the next iteration
                            # will fold them into one render at burst end.
                            if not _pending_buf:
                                try:
                                    rs, _, _ = select.select([sys.stdin], [], [], 0)
                                except Exception:
                                    rs = []
                                if not rs:
                                    do_render()
                            continue
                        if ch in ("__DOWN__", "__WHEEL_DOWN__"):
                            palette_sel += 1   # render clamps to sugg length
                            if not _pending_buf:
                                try:
                                    rs, _, _ = select.select([sys.stdin], [], [], 0)
                                except Exception:
                                    rs = []
                                if not rs:
                                    do_render()
                            continue
                        if ch in ("\r", "\n"):
                            cline = cmd_input.strip()
                            cmd_input = None
                            try:
                                lines = run_palette_cmd(cline)
                            except KeyboardInterrupt:
                                raise
                            except Exception as exc:
                                lines = [f"error: {type(exc).__name__}: {exc}"]
                            last_cmd_lines = lines or []
                            # 5-min safety fade; the next nav-key dismisses
                            # the output immediately (see below).
                            last_cmd_until = time.time() + 300.0
                            panel_scroll = 0    # new output → start at top
                            break
                        if ch in ("\x7f", "\b"):   # backspace
                            cmd_input = cmd_input[:-1]
                            if cmd_input == "":
                                palette_show_all = False
                            palette_sel = 0    # filtered list changed
                            do_render()
                            continue
                        if ch == "\t":             # Tab — fill in selected row
                            if cmd_input == "":
                                palette_show_all = True
                            before = cmd_input
                            sugg = _palette_completions(cmd_input)
                            if sugg:
                                sel = max(0, min(palette_sel, len(sugg) - 1))
                                value = sugg[sel][0]
                                tparts = cmd_input.split(" ")
                                if len(tparts) == 1:
                                    # Command-name suggestions carry a leading
                                    # `/`; cmd_input doesn't include the slash.
                                    cmd_input = value.lstrip("/") + " "
                                else:
                                    tparts[-1] = value
                                    cmd_input = " ".join(tparts)
                                    if not value.endswith("="):
                                        cmd_input += " "
                                palette_sel = 0    # reset for the new context
                            # If Tab made no progress and the buffer is a
                            # non-empty command, treat as Enter — run it.
                            if cmd_input == before and cmd_input.strip():
                                cline = cmd_input.strip()
                                cmd_input = None
                                try:
                                    lines = run_palette_cmd(cline)
                                except KeyboardInterrupt:
                                    raise
                                except Exception as exc:
                                    lines = [f"error: {type(exc).__name__}: {exc}"]
                                last_cmd_lines = lines or []
                                last_cmd_until = time.time() + 300.0
                                break
                            do_render()
                            continue
                        if ch.isprintable() and len(cmd_input) < 200:
                            cmd_input += ch
                            palette_sel = 0    # filtered list changed
                            do_render()
                            continue
                        # ignore other control chars
                    do_render()
                    # Force the outer loop to refresh state immediately so
                    # the effects of the command (cancel/release/timeout/…)
                    # are visible right away.
                    deadline = 0
                    continue

                # Scroll keys never dismiss palette output.  They scroll
                # the PANEL only when its content overflows the available
                # rows (think `/help`); for short outputs (like the
                # 1-line "view: runtimes (all)" confirmation) they fall
                # through to scroll the underlying view, and the panel
                # stays visible.  Non-scroll keys still dismiss as before.
                _SCROLL_KEYS = ("__UP__", "__DOWN__", "__PGUP__", "__PGDN__",
                                "__HOME__", "__END__", "__WHEEL_UP__", "__WHEEL_DOWN__")
                output_visible_now = bool(last_cmd_lines and time.time() < last_cmd_until)
                if key in _SCROLL_KEYS and output_visible_now:
                    try:
                        term_rows_now = shutil.get_terminal_size().lines
                    except Exception:
                        term_rows_now = 30
                    overlay_cap = max(0, term_rows_now - 4)
                    if len(last_cmd_lines) > overlay_cap:
                        # Long output (e.g. /help) — scroll the panel.
                        if   key in ("__UP__", "__WHEEL_UP__"):   panel_scroll = max(0, panel_scroll - 1)
                        elif key in ("__DOWN__", "__WHEEL_DOWN__"): panel_scroll += 1
                        elif key == "__PGUP__": panel_scroll = max(0, panel_scroll - 10)
                        elif key == "__PGDN__": panel_scroll += 10
                        elif key == "__HOME__": panel_scroll = 0
                        elif key == "__END__":  panel_scroll = 10**9
                        do_render()
                        continue
                    # else: fall through to view-scroll handler below.
                    # Don't dismiss the panel — the user is scrolling, not
                    # navigating away.
                # Whitelist the keys that dismiss the panel: anything that
                # changes focus / view (view-switch shortcuts + ESC/SPACE
                # for overview + `a` for the all-toggle).  `/` re-enters
                # the palette and naturally replaces the panel content
                # with whatever the new command returns.  Scroll keys,
                # mouse events, and stray bytes from focus / X10-mouse
                # reports all leave the panel alone.
                # ESC is intentionally NOT in this set: arrow-key sequences
                # sometimes race the lone-ESC detector and a misdetect would
                # then dismiss the panel mid-scroll.  ESC still changes
                # view to overview via the dispatch below.
                _DISMISS_KEYS = {"o", "r", "l", "j", "c", "f", "e", "a", " "}
                if key in _DISMISS_KEYS and output_visible_now:
                    last_cmd_lines = []
                    last_cmd_until = 0.0
                    panel_scroll = 0
                    output_was_visible = True
                else:
                    output_was_visible = False

                prev_view = view
                prev_view_all = view_all
                # ESC is intentionally a no-op at the dashboard level.
                # Arrow-key sequences sometimes race the lone-ESC detector
                # in `_read_key`, so a misdetected arrow used to dismiss
                # the panel or yank back to overview after a single
                # successful scroll.  Now misdetections are silent.
                # Use `o` or SPACE for overview.
                if key in ("o", " "):    # overview (`o` or SPACE)
                    # ESC is no longer mapped — arrow-key sequences can race
                    # the lone-ESC detector and were yanking the user back
                    # to overview every time.  Use `o` or SPACE.
                    view = "overview"
                elif key in ("i", "l"):
                    # `i` = instances (= runtimes view internally; the
                    # palette command keeps `/runtimes` for backward
                    # compatibility, only the user-facing label changed).
                    # `l` is preserved as an alias for muscle memory.
                    # `r` is now bound to reload (below) — the only fix
                    # for a stuck "⚠ network" banner used to be Ctrl+C
                    # and restart.
                    view = "runtimes"
                elif key == "r":
                    _reload_dashboard()
                elif key == "j":
                    view = "jobs"
                elif key == "c":
                    view = "cost"
                elif key == "f":
                    view = "follow"
                elif key == "e":
                    view = "events"
                elif key == "a" and view in ("jobs", "runtimes"):
                    # Toggle "all" (include terminal jobs / released runtimes).
                    view_all = not view_all
                    # Content size changes drastically — reset scroll.
                    scroll_offsets[view] = 0
                    at_tail[view] = False
                elif key in ("__UP__", "__WHEEL_UP__") and view in scrollable_views:
                    scroll_offsets[view] = max(0, scroll_offsets[view] - 1)
                    at_tail[view] = False
                elif key in ("__DOWN__", "__WHEEL_DOWN__") and view in scrollable_views:
                    scroll_offsets[view] = scroll_offsets[view] + 1
                    # Re-tail flag is set in do_render once we know max_off.
                    at_tail[view] = False
                elif key == "__PGUP__" and view in scrollable_views:
                    scroll_offsets[view] = max(0, scroll_offsets[view] - 10)
                    at_tail[view] = False
                elif key == "__PGDN__" and view in scrollable_views:
                    scroll_offsets[view] = scroll_offsets[view] + 10
                    at_tail[view] = False
                elif key == "__HOME__" and view in scrollable_views:
                    scroll_offsets[view] = 0
                    at_tail[view] = False
                elif key == "__END__" and view in scrollable_views:
                    at_tail[view] = True    # snaps to bottom on next render
                else:
                    if output_was_visible:
                        do_render()
                    continue
                if view != prev_view or view_all != prev_view_all or output_was_visible:
                    # Reset view_all when leaving its applicable views — no
                    # invisible state lurking when the user comes back.
                    if view not in ("jobs", "runtimes"):
                        view_all = False
                    do_render()
                    continue
                # Scroll-only update (view didn't change): coalesce
                # bursts of arrow / wheel events into a single render.
                # If more keystrokes are already queued (in _pending_buf
                # or sitting in stdin), skip this render — the next
                # iteration will fold those updates into the same final
                # state.
                if _pending_buf:
                    continue
                try:
                    rsel_peek, _, _ = select.select([sys.stdin], [], [], 0)
                except Exception:
                    rsel_peek = []
                if rsel_peek:
                    continue
                do_render()
    except KeyboardInterrupt:
        pass
    finally:
        if isatty and old_term is not None:
            try: termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
            except Exception: pass
        # Restore cursor + leave alt screen → user's previous scrollback returns.
        sys.stdout.write(SHOW + WRAP_ON + ALT_OFF); sys.stdout.flush()
        print("\033[36m●  live dashboard stopped.\033[0m")
    return 0


def _wait_for_runtime(query: str) -> int:
    """Block until the named runtime is released."""
    sess = make_session()
    try:
        eps = _resolve_endpoints(query, list_runtimes(sess))
    except ValueError as e:
        print(f"[direct_kernel] {e}", file=sys.stderr)
        return 2
    if len(eps) != 1:
        print(f"[direct_kernel] {query!r} matches multiple runtimes; pick one",
              file=sys.stderr)
        return 2
    return _watch_events(types={"runtime_released"}, endpoint=eps[0], once=True)


# ── runtime / accelerator label helpers ───────────────────────────────────────

def _runtime_letter(idx: int) -> str:
    """0→a, 1→b, …, 25→z, 26→aa, 27→ab, 51→az, 52→ba, …"""
    if idx < 26:
        return chr(ord("a") + idx)
    return _runtime_letter(idx // 26 - 1) + chr(ord("a") + idx % 26)


def _heal_stale_released(runtimes: list[dict]) -> None:
    """Clear `released_at` / `released_by` / `preempted` on any endpoint
    whose meta says "released" but which is actually in the live API list.

    Repairs corruption from the pre-fix false-preemption bug, and protects
    against any future ones.  Cheap (one file read per active endpoint).
    """
    for r in runtimes:
        ep = r["endpoint"]
        m  = _runtime_meta(ep) or {}
        if m.get("released_at"):
            try:
                _set_runtime_meta(ep,
                                  released_at=None,
                                  released_by=None,
                                  preempted=None)
            except Exception:
                pass


def _stamp_missing_letters(runtimes: list[dict]) -> None:
    """Ensure every runtime in `runtimes` has a stable letter stamped in meta.

    Reads existing letters from meta first.  For any runtime without one
    (legacy entries, externally-assigned runtimes from web Colab, etc.),
    assigns the lowest letter NOT currently used by any other active
    runtime, stamps it back to meta, and continues.

    This is the source of letter stability: once a runtime has a letter, it
    keeps it for its entire lifetime — even if other runtimes get assigned
    or released around it.  After a runtime is released, its letter slot
    becomes available for the next new runtime.
    """
    used: set[str] = set()
    unmapped: list[str] = []
    for r in runtimes:
        ep = r["endpoint"]
        m = _runtime_meta(ep) or {}
        if m.get("letter"):
            used.add(m["letter"])
        else:
            unmapped.append(ep)
    for ep in sorted(unmapped):
        i = 0
        while _runtime_letter(i) in used:
            i += 1
        ltr = _runtime_letter(i)
        try:
            _set_runtime_meta(ep, letter=ltr)
        except Exception:
            pass
        used.add(ltr)


def _shorthand_map(runtimes: list[dict]) -> dict[str, str]:
    """Stable letter → endpoint map for the currently-active runtimes.

    Reads pre-stamped `meta.letter` for each runtime.  Falls back to a
    deterministic alphabetical assignment for endpoints that don't have a
    stamped letter yet (read-only fallback — call ``_stamp_missing_letters``
    first if you want the assignment persisted).
    """
    out: dict[str, str] = {}
    unmapped: list[str] = []
    for r in runtimes:
        ep  = r["endpoint"]
        ltr = (_runtime_meta(ep) or {}).get("letter")
        if ltr and ltr not in out:
            out[ltr] = ep
        else:
            unmapped.append(ep)
    used = set(out.keys())
    for ep in sorted(unmapped):
        i = 0
        while _runtime_letter(i) in used:
            i += 1
        ltr = _runtime_letter(i)
        out[ltr] = ep
        used.add(ltr)
    return out


def _normalized_accel(r: dict) -> str:
    ac = r.get("accelerator", "")
    return "CPU" if ac in ("NONE", "VARIANT_UNSPECIFIED", "") else ac


# ── cost / billing ────────────────────────────────────────────────────────────

# Approximate Colab Pro/Pro+ accelerator rates in compute-units-per-hour for
# the STANDARD (low-RAM) machine shape.  Not published by Colab as an API.
# High-RAM variants (`--high-ram`) appear to cost roughly 1.4–2× the standard
# rate.  `--cost` cross-references against the live `paidComputeUnitsBalance`
# so the "Observed rate" line is what you should trust if these drift.
_COST_RATE_TABLE: dict[str, float] = {
    "CPU":  0.1,
    "T4":   1.84,
    "L4":   2.85,
    "A100": 5.50,
    "H100": 8.00,    # rough; H100 is rarely-seen on Pro/Pro+
    "G4":   2.00,
    "V5E1": 1.00,
    "V6E1": 1.50,
}
_HIGH_RAM_MULT = 1.5    # rough multiplier; only applied if the runtime record says SHAPE_HIGH_MEM

_BALANCE_LOG = _JOBS_DIR / "balance.jsonl"
_BALANCE_KEEP_HOURS = 6.0     # how much history we keep for the observed-rate calc


def _account_balance(sess: requests.Session) -> float | None:
    """Live `paidComputeUnitsBalance` from /v1/user-info, or None on error."""
    try:
        r = sess.get(f"{COLAB_GAPI_DOMAIN}/v1/user-info", timeout=15)
        r.raise_for_status()
        return float(r.json().get("paidComputeUnitsBalance", 0.0))
    except Exception:
        return None


def _record_balance(balance: float) -> None:
    """Append (now, balance) to balance.jsonl, prune entries older than ~6h."""
    try:
        _JOBS_DIR.mkdir(exist_ok=True)
        with _BALANCE_LOG.open("a") as f:
            f.write(json.dumps({"t": time.time(), "balance": balance}) + "\n")
        # Periodic prune (cheap).
        cutoff = time.time() - _BALANCE_KEEP_HOURS * 3600
        keep = []
        for ln in _BALANCE_LOG.read_text().splitlines():
            try:
                rec = json.loads(ln)
                if rec.get("t", 0) >= cutoff:
                    keep.append(ln)
            except Exception:
                pass
        # Only rewrite if we actually pruned anything (to avoid contention).
        if len(keep) > 0 and len(keep) < sum(1 for _ in _BALANCE_LOG.open()):
            _BALANCE_LOG.write_text("\n".join(keep) + "\n")
    except Exception:
        pass


def _observed_rate_per_hour() -> float | None:
    """Compute units-per-hour from balance.jsonl samples within the last
    ~30 min.  Falls back to a wider window if we don't have enough recent
    samples.  Returns None if we can't span >5 minutes either way.

    Why short window: if we average across the full 6-hour log, idle periods
    dilute the rate.  A user actively running an A100 expects to see ~5.5/h,
    not the time-weighted average across hours of nothing.
    """
    if not _BALANCE_LOG.exists():
        return None
    try:
        samples = [json.loads(ln) for ln in _BALANCE_LOG.read_text().splitlines() if ln.strip()]
    except Exception:
        return None
    if len(samples) < 2:
        return None
    samples.sort(key=lambda r: r["t"])
    now = time.time()

    def _rate(window_s: float) -> float | None:
        recent = [s for s in samples if now - s["t"] <= window_s]
        if len(recent) < 2:
            return None
        s0, s1 = recent[0], recent[-1]
        dt_h = (s1["t"] - s0["t"]) / 3600
        if dt_h < (5 / 60):
            return None
        return (s0["balance"] - s1["balance"]) / dt_h

    # Try short window first, fall back to longer if we don't have enough.
    return _rate(30 * 60) or _rate(2 * 3600) or _rate(_BALANCE_KEEP_HOURS * 3600)


def _runtime_cost(r: dict) -> tuple[float | None, float | None, float | None]:
    """Return (rate_per_hour, hours_running, est_cost) for a runtime record."""
    accel = _normalized_accel(r)
    rate  = _COST_RATE_TABLE.get(accel)
    if rate is not None and (r.get("machineShape", "") or "").upper() == "SHAPE_HIGH_MEM":
        rate *= _HIGH_RAM_MULT
    meta  = _runtime_meta(r["endpoint"]) or {}
    assigned = meta.get("assigned_at")
    if assigned is None:
        return rate, None, None
    hrs = (time.time() - assigned) / 3600
    cost = rate * hrs if rate is not None else None
    return rate, hrs, cost


def _print_cost_table(rts: list[dict], balance: float | None, observed_rate: float | None) -> None:
    if balance is not None:
        print(f"Balance:  {balance:.2f} compute units remaining")
    else:
        print("Balance:  (could not fetch)")
    if not rts:
        print("\nNo active runtimes — no burn.")
        return
    sh_map = _shorthand_map(rts)
    inv    = {ep: sh for sh, ep in sh_map.items()}
    rts    = sorted(rts, key=lambda r: r["endpoint"])

    print(f"\n{'SH':<3} {'ENDPOINT':<38} {'ACCEL':<6} {'UPTIME':<8} {'RATE/h':<10} {'EST COST':<10} {'DESC':<26}")
    print("─" * 106)
    total_rate = 0.0
    total_cost = 0.0
    for r in rts:
        ep    = r["endpoint"]
        sh    = inv.get(ep, "?")
        accel = _normalized_accel(r)
        rate, hrs, cost = _runtime_cost(r)
        rate_s  = f"{rate:.2f}" if rate is not None else "?"
        hrs_s   = f"{hrs:.2f}h" if hrs is not None else "?"
        cost_s  = f"{cost:.2f} u" if cost is not None else "?"
        desc    = _trunc((_runtime_meta(ep) or {}).get("desc"), 25) or "—"
        print(f"{sh:<3} {ep[:38]:<38} {accel:<6} {hrs_s:<8} {rate_s:<10} {cost_s:<10} {desc:<26}")
        if rate: total_rate += rate
        if cost: total_cost += cost
    print("─" * 106)
    print(f"{'':<3} {'TOTAL':<38} {'':<6} {'':<8} {f'{total_rate:.2f}':<10} {f'{total_cost:.2f} u':<10}")
    print(f"\nEstimated burn right now: {total_rate:.2f} units/hour")
    if observed_rate is not None:
        sign = "burning" if observed_rate >= 0 else "(net gain — odd)"
        print(f"Observed rate (from balance log): {observed_rate:.2f} units/hour  {sign}")
    else:
        print("Observed rate: not enough history yet — re-run `colab --cost` "
              "after a few minutes to see the live rate.")


# ── runtime management ─────────────────────────────────────────────────────────

def list_runtimes(sess: requests.Session) -> list[dict]:
    """List all active Colab runtimes on the account."""
    r = sess.get(f"{COLAB_GAPI_DOMAIN}/v1/assignments", timeout=15)
    r.raise_for_status()
    return r.json().get("assignments", [])


def unassign_runtime(sess: requests.Session, endpoint: str,
                     *, released_by: str = "user") -> dict:
    """
    Release a Colab runtime by its endpoint name.

    ``released_by`` is recorded in the runtime metadata for later display
    (e.g. in ``--follow``'s "runtime released" banner).  Common values:
      * ``"user"`` (default) — explicit ``colab --unassign``.
      * ``"idle_timeout"`` — the reaper unassigning after the idle deadline.
      * ``"preempted"`` — set directly by the preemption-detect path
        (skips this function altogether).
      * ``"already_released"`` — set automatically when the API returns 404
        (something else already killed it).

    Same two-step xsrf pattern as assign:
      GET  /tun/m/unassign/{endpoint}?authuser=0  → {token (xsrf)}
      POST /tun/m/unassign/{endpoint}?authuser=0  with X-Goog-Colab-Token  → 204
    """
    url    = f"{COLAB_DOMAIN}/tun/m/unassign/{endpoint}"
    params = {"authuser": "0"}
    hdrs   = {"X-Colab-Tunnel": "Google"}

    def _stamp(reason: str) -> None:
        try:
            _set_runtime_meta(endpoint, released_at=time.time(), released_by=reason)
            _emit_event("runtime_released", endpoint=endpoint, reason=reason)
        except Exception:
            pass

    r = sess.get(url, params=params, headers=hdrs, timeout=15)
    if r.status_code == 404:
        _stamp("already_released")
        return {"status": "already_released", "endpoint": endpoint}
    r.raise_for_status()
    xsrf = _xssi_json(r.text)["token"]

    r = sess.post(url, params=params, headers={**hdrs, "X-Goog-Colab-Token": xsrf}, timeout=15)
    if r.status_code == 404:
        _stamp("already_released")
        return {"status": "already_released", "endpoint": endpoint}
    r.raise_for_status()
    # Don't delete the metadata — leave it so `colab --runtimes --all` and
    # historical cost queries can still see this runtime existed.  Just stamp
    # released_at + released_by so we know who killed it.
    _stamp(released_by)
    return {"status": "unassigned", "endpoint": endpoint, "released_by": released_by}


def _create_throwaway_notebook(sess: requests.Session) -> str:
    """Create a fresh `.ipynb` on Drive and return its file_id.

    Used for multi-runtime provisioning: Colab enforces 1:1 between a notebook
    and a runtime, so to have N concurrent runtimes the bridge has to own N
    notebooks.  This creates one on demand.  The file isn't tracked anywhere
    on disk — once the runtime is provisioned the notebook isn't needed for
    submit / cancel / unassign (those use the endpoint, not the notebook).
    """
    print("[direct_kernel] Creating throwaway notebook on Drive…", flush=True)
    data = _drive_multipart_upload(
        sess,
        metadata={
            "name":     "direct_kernel_runtime_pool",
            "mimeType": "application/vnd.google.colaboratory",
        },
        body=_NOTEBOOK_TEMPLATE,
    )
    fid = data["id"]
    print(f"[direct_kernel]   notebook {fid}", flush=True)
    return fid


def assign_runtime(
    sess: requests.Session,
    file_id: str,
    accelerator: str = "A100",
    high_ram: bool = False,
    idle_timeout_min: int | None = _DEFAULT_IDLE_TIMEOUT_MIN,
    allow_pool: bool = False,
    desc: str | None = None,
) -> tuple[str, str, str]:
    """
    Provision a Colab runtime.  Returns (jupyter_url, proxy_token, endpoint).

    Protocol (from VS Code extension assign() method):
      GET  /tun/m/assign?nbh=...  → {token (xsrf), ...}
      POST /tun/m/assign          → {endpoint, runtimeProxyInfo: {url, token, tokenExpiresInSeconds}}
    """
    nbh           = _notebook_hash(file_id)
    url           = f"{COLAB_DOMAIN}{ASSIGN_PATH}"
    accel_q, var_q = _ACCEL_MAP.get(
        accelerator.upper(), (accelerator.upper(), "GPU")
    )
    params: dict = {"nbh": nbh, "authuser": "0"}
    if accel_q is not None:
        params["accelerator"] = accel_q
    if var_q is not None:
        params["variant"] = var_q
    if high_ram:
        params["shape"] = "SHAPE_HIGH_MEM"

    tunnel_hdr = {"X-Colab-Tunnel": "Google"}

    # Step 1: GET → xsrf token
    print(f"[direct_kernel] GET /assign (accelerator={accelerator})…", flush=True)
    r = sess.get(url, params=params, headers=tunnel_hdr, timeout=30)
    if r.status_code == 401:
        raise RuntimeError(
            "Auth rejected (401).  Run: python3 colab_bridge/direct_kernel.py --auth"
        )
    r.raise_for_status()
    data = _xssi_json(r.text)

    if data.get("p"):
        raise RuntimeError(
            "Colab requires reCAPTCHA.  Open the notebook in a browser once to clear it."
        )

    # If GET returned runtimeProxyInfo directly, the notebook already has an assigned
    # runtime.  Two paths from here:
    #   * allow_pool=False (default, e.g. `colab -c "..."`) — silently reuse it.
    #   * allow_pool=True  (e.g. `colab --assign`)         — create a fresh
    #     throwaway notebook and provision a NEW runtime there, so the user
    #     gets the additional runtime they asked for.
    if "runtimeProxyInfo" in data:
        if allow_pool:
            print(f"[direct_kernel] Primary notebook already bound — using a fresh "
                  f"throwaway notebook to provision an additional runtime.", flush=True)
            new_fid = _create_throwaway_notebook(sess)
            return assign_runtime(sess, new_fid, accelerator, high_ram,
                                  idle_timeout_min=idle_timeout_min,
                                  allow_pool=False)

        proxy_info  = data["runtimeProxyInfo"]
        jupyter_url = proxy_info["url"].rstrip("/")
        proxy_token = proxy_info["token"]
        endpoint    = data.get("endpoint", jupyter_url.split("//")[1].split("-b.")[0].split("-c.")[0])
        print(f"[direct_kernel] Runtime already assigned → {endpoint}", flush=True)
        # Update timeout if explicitly given; create meta record if missing.
        existing = _runtime_meta(endpoint) or {}
        fields   = {"accelerator": accelerator.upper()}
        if "assigned_at" not in existing:
            fields["assigned_at"] = time.time()
        if idle_timeout_min is not None:
            fields["idle_timeout_min"] = idle_timeout_min
        elif "idle_timeout_min" not in existing:
            fields["idle_timeout_min"] = _DEFAULT_IDLE_TIMEOUT_MIN
        if desc is not None:
            fields["desc"] = desc
        # Always refresh region (cheap, sometimes useful if the proxy changed).
        fields["region"] = _parse_region_from_url(jupyter_url)
        # Clear any stale lifecycle fields from a previous run on this endpoint.
        fields["released_at"] = None
        fields["released_by"] = None
        fields["preempted"]   = None
        _set_runtime_meta(endpoint, **fields)
        _emit_event("runtime_assigned",
                    endpoint=endpoint,
                    accel=accelerator.upper(),
                    region=_parse_region_from_url(jupyter_url),
                    reused=True)
        try:
            _spawn_reaper(endpoint)
        except Exception:
            pass
        return jupyter_url, proxy_token, endpoint

    if "token" not in data:
        raise RuntimeError(
            f"Unexpected GET /assign response (no token or runtimeProxyInfo): {data!r}"
        )

    xsrf = data["token"]

    # Step 2: POST → runtime info (can take 30-120s)
    with _spin("provisioning runtime"):
        r = sess.post(url, params=params, headers={**tunnel_hdr, "X-Goog-Colab-Token": xsrf}, timeout=120)
    if r.status_code == 503:
        raise RuntimeError("No runtime available right now.  Try again shortly.")
    if r.status_code == 412:
        raise RuntimeError("Insufficient Colab quota.")
    r.raise_for_status()

    data        = _xssi_json(r.text)
    proxy_info  = data["runtimeProxyInfo"]
    jupyter_url = proxy_info["url"].rstrip("/")
    proxy_token = proxy_info["token"]
    endpoint    = data["endpoint"]
    expires_in  = proxy_info.get("tokenExpiresInSeconds", "?")
    print(f"[direct_kernel] Runtime ready → {endpoint}  (token expires {expires_in}s)", flush=True)
    # Pick a stable letter for this runtime — lowest one not used by any
    # currently-active runtime.  Stamped into meta so it never shifts.
    try:
        live_eps = [r["endpoint"] for r in list_runtimes(sess)
                    if r["endpoint"] != endpoint]
    except Exception:
        live_eps = []
    used_letters = set()
    for ep_ in live_eps:
        m = _runtime_meta(ep_) or {}
        if m.get("letter"):
            used_letters.add(m["letter"])
    i = 0
    while _runtime_letter(i) in used_letters:
        i += 1
    new_letter = _runtime_letter(i)

    _set_runtime_meta(
        endpoint,
        accelerator      = accelerator.upper(),
        assigned_at      = time.time(),
        region           = _parse_region_from_url(jupyter_url),
        idle_timeout_min = (idle_timeout_min if idle_timeout_min is not None
                            else _DEFAULT_IDLE_TIMEOUT_MIN),
        desc             = desc,
        letter           = new_letter,
        # Clear stale lifecycle fields if this endpoint was ever recorded before.
        released_at      = None,
        released_by      = None,
        preempted        = None,
    )
    _emit_event("runtime_assigned",
                endpoint=endpoint,
                accel=accelerator.upper(),
                region=_parse_region_from_url(jupyter_url))
    try:
        _spawn_reaper(endpoint)
    except Exception:
        pass
    return jupyter_url, proxy_token, endpoint


# ── proxy headers for Jupyter calls ───────────────────────────────────────────

def _proxy_hdrs(proxy_token: str) -> dict:
    return {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Tunnel":              "Google",
    }


# ── kernel management ──────────────────────────────────────────────────────────

def get_or_create_kernel(
    sess: requests.Session,
    jupyter_url: str,
    proxy_token: str,
) -> str:
    ph = _proxy_hdrs(proxy_token)
    r  = sess.get(f"{jupyter_url}/api/kernels", headers=ph, timeout=20)
    r.raise_for_status()
    kernels = r.json()
    if kernels:
        kid = kernels[0]["id"]
        print(f"[direct_kernel] Using kernel {kid}", flush=True)
        return kid
    print("[direct_kernel] Creating kernel…", flush=True)
    r = sess.post(
        f"{jupyter_url}/api/kernels",
        headers=ph, json={"name": "python3"}, timeout=30,
    )
    r.raise_for_status()
    kid = r.json()["id"]
    print(f"[direct_kernel] Created kernel {kid}", flush=True)
    with _spin("starting kernel"):
        time.sleep(3)
    return kid


def _do_interrupt(sess: requests.Session, jupyter_url: str, proxy_token: str, kernel_id: str) -> bool:
    try:
        sess.post(
            f"{jupyter_url}/api/kernels/{kernel_id}/interrupt",
            headers=_proxy_hdrs(proxy_token), timeout=10,
        ).raise_for_status()
        return True
    except Exception as e:
        print(f"[direct_kernel] interrupt request: {e}", flush=True)
        return False


def _do_restart(sess: requests.Session, jupyter_url: str, proxy_token: str, kernel_id: str) -> str:
    """Restart kernel in-place; returns same kernel_id."""
    sess.post(
        f"{jupyter_url}/api/kernels/{kernel_id}/restart",
        headers=_proxy_hdrs(proxy_token), timeout=30,
    ).raise_for_status()
    print("[direct_kernel] Kernel restarted.", flush=True)
    return kernel_id


def _do_force_restart(sess: requests.Session, jupyter_url: str, proxy_token: str, kernel_id: str) -> str:
    """Kill the kernel and create a fresh one.  Returns new kernel_id."""
    print("[direct_kernel] Force-killing kernel…", flush=True)
    try:
        sess.delete(
            f"{jupyter_url}/api/kernels/{kernel_id}",
            headers=_proxy_hdrs(proxy_token), timeout=10,
        )
    except Exception as e:
        print(f"[direct_kernel] kill: {e}", flush=True)
    time.sleep(1)
    return get_or_create_kernel(sess, jupyter_url, proxy_token)


# ── keep-alive ─────────────────────────────────────────────────────────────────

def _keepalive_loop(
    sess: requests.Session,
    endpoint: str,
    stop: threading.Event,
    interval: float = 60.0,
) -> None:
    """Pings /tun/m/{endpoint}/keep-alive/ every `interval` seconds until
    stopped.  Quiet by default — only prints if the endpoint returns 404/410
    (server-side teardown / preemption), in which case it stamps
    ``released_at`` + ``preempted`` and stops itself.
    """
    ka_url = f"{COLAB_DOMAIN}/tun/m/{endpoint}/keep-alive/"
    while not stop.wait(interval):
        _ensure_fresh_token(sess)
        try:
            r = sess.get(ka_url, headers={"X-Colab-Tunnel": "Google"}, timeout=10)
            if r.status_code in (400, 404, 410):
                # 4xx is suspicious but not definitive — also seen during
                # warmup or transient blips.  Confirm via the strict
                # double-check before stamping preempted.
                if not _is_preempted_double_check(sess, endpoint):
                    continue   # transient — keep going
                print(f"\n[direct_kernel] !! runtime {endpoint} appears preempted "
                      f"(keep-alive {r.status_code}, missing from list_runtimes "
                      f"on double-check). In-flight job will likely fail.",
                      file=sys.stderr, flush=True)
                try:
                    if (_runtime_meta(endpoint) or {}).get("released_at"):
                        return    # already released by another path
                    _set_runtime_meta(endpoint,
                                      released_at=time.time(),
                                      released_by="preempted",
                                      preempted=True)
                    _emit_event("runtime_released", endpoint=endpoint, reason="preempted")
                    _reap_log(endpoint, "PREEMPTED — keep-alive (foreground) "
                                        f"returned {r.status_code}, confirmed via double-check")
                except Exception:
                    pass
                return
        except Exception:
            pass    # transient errors — keep going


# ── job persistence (file-backed registry) ─────────────────────────────────────

class _JobStore:
    """File-backed registry of every job submitted via this tool.

    Layout under ``BRIDGE_DIR/.direct_kernel_jobs/``:

      * ``index.json``                — locked list of job records
      * ``<job_id>.events.jsonl``     — line-delimited events for one job

    The index is updated under ``fcntl.flock`` so concurrent CLI invocations
    don't trample each other.  Per-job event files are append-only so readers
    can ``seek``+``tail`` them without coordination.
    """

    def __init__(self, root: Path = _JOBS_DIR) -> None:
        self.root       = root
        self.index_path = self.root / "index.json"
        self.root.mkdir(exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("[]")

    def events_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.events.jsonl"

    @contextlib.contextmanager
    def _lock(self):
        import fcntl
        with self.index_path.open("r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield f
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def list_jobs(self) -> list[dict]:
        # Acquire a shared flock so we can't race a writer mid-truncate.
        # Without this, a concurrent `update_job` (which truncates +
        # rewrites under LOCK_EX) can leave the file briefly empty;
        # an unlocked reader then sees "" → JSONDecodeError → [], and
        # callers like `_internal_reap` and `_runtime_has_active_jobs`
        # silently see "no jobs" and reap a runtime that had recent
        # activity.  LOCK_SH is non-exclusive among readers and just
        # waits for any in-progress LOCK_EX writer to finish.
        import fcntl
        try:
            with self.index_path.open("r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    txt = f.read()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return json.loads(txt or "[]")
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_job(self, job_id: str) -> dict | None:
        for rec in self.list_jobs():
            if rec["job_id"] == job_id:
                return rec
        return None

    def add_job(self, rec: dict) -> None:
        with self._lock() as f:
            f.seek(0)
            data = json.loads(f.read() or "[]")
            data.append(rec)
            f.seek(0); f.truncate()
            json.dump(data, f, indent=2)
        try:
            _emit_event("job_queued",
                        jid=rec.get("job_id"),
                        endpoint=rec.get("endpoint"),
                        accel=rec.get("accelerator"))
        except Exception:
            pass

    def update_job(self, job_id: str, **fields) -> dict | None:
        with self._lock() as f:
            f.seek(0)
            data = json.loads(f.read() or "[]")
            updated: dict | None = None
            for rec in data:
                if rec["job_id"] == job_id:
                    rec.update(fields)
                    updated = rec
                    break
            if updated is not None:
                f.seek(0); f.truncate()
                json.dump(data, f, indent=2)
            return updated

    def append_event(self, job_id: str, event: dict) -> None:
        with self.events_path(job_id).open("a") as f:
            f.write(json.dumps(event) + "\n")

    def latest_job(self) -> dict | None:
        """The most recent job that actually started running (skips queued)."""
        jobs = [j for j in self.list_jobs() if j.get("status") != "queued"]
        if not jobs:
            return None
        return max(jobs, key=lambda r: r.get("started", 0))


def _stream_from_disk(
    store: _JobStore,
    job_id: str,
    *,
    show: bool   = True,
    since: int   = 0,
    poll: float  = 0.15,
) -> Iterator[dict]:
    """Replay history from events.jsonl, then tail until job leaves running."""
    path = store.events_path(job_id)
    rec  = store.get_job(job_id)
    if rec is None:
        raise KeyError(f"Unknown job_id: {job_id!r}")

    seen = 0
    while True:
        rec = store.get_job(job_id) or rec
        if path.exists():
            with path.open() as f:
                lines = [ln for ln in f if ln.strip()]
            for ev in (json.loads(ln) for ln in lines[seen:]):
                if show:
                    text = ev.get("text", "")
                    if ev.get("type") in ("stderr", "error"):
                        sys.stderr.write(text); sys.stderr.flush()
                    else:
                        sys.stdout.write(text); sys.stdout.flush()
                if seen >= since:
                    yield ev
                seen += 1
        if rec.get("status") not in ("queued", "running"):
            return
        time.sleep(poll)


# ── job model ──────────────────────────────────────────────────────────────────

class _Job:
    """Single code-execution job.  Thread-safe event store."""

    __slots__ = ("job_id", "code", "status", "events", "started", "ended",
                 "_lock", "_done", "_store")

    def __init__(self, job_id: str, code: str, store: _JobStore | None = None) -> None:
        self.job_id  = job_id
        self.code    = code
        self.status  = "pending"
        self.events: list[dict] = []
        self.started = time.time()
        self.ended: float | None = None
        self._lock   = threading.Lock()
        self._done   = threading.Event()
        self._store  = store

    def _append(self, ev: dict) -> None:
        with self._lock:
            self.events.append(ev)
        if self._store is not None:
            try:
                self._store.append_event(self.job_id, ev)
            except Exception:
                pass   # disk persistence must never break execution

    def _finish(self, status: str = "done") -> None:
        # Preserve a previously-set "cancelled" status so a --cancel signal
        # is never overwritten by the worker's natural completion path.
        if self._store is not None:
            try:
                existing = self._store.get_job(self.job_id)
                if existing and existing.get("status") == "cancelled":
                    status = "cancelled"
            except Exception:
                pass
        with self._lock:
            self.status = status
            self.ended  = time.time()
        if self._store is not None:
            try:
                self._store.update_job(self.job_id, status=status, ended=self.ended)
                rec = self._store.get_job(self.job_id) or {}
                _emit_event(f"job_{status}",
                            jid=self.job_id,
                            endpoint=rec.get("endpoint"),
                            accel=rec.get("accelerator"),
                            elapsed_s=round(self.ended - self.started, 1))
            except Exception:
                pass
        self._done.set()

    def iter_events(self, since: int = 0, poll: float = 0.15) -> Iterator[dict]:
        """Yield events as they arrive; blocks until done."""
        ptr = since
        while True:
            with self._lock:
                chunk = self.events[ptr:]
                done  = self._done.is_set()
            for ev in chunk:
                yield ev
                ptr += 1
            if done and ptr >= (len(self.events)):
                break
            if not chunk:
                time.sleep(poll)

    def summary(self) -> dict:
        elapsed = (self.ended or time.time()) - self.started
        return {
            "job_id":  self.job_id,
            "status":  self.status,
            "events":  len(self.events),
            "elapsed": round(elapsed, 1),
        }


# ── synchronous WebSocket executor ────────────────────────────────────────────
#
# asyncio.open_connection() fails on this machine when Tailscale is active
# (non-blocking sockets interact differently with Tailscale's virtual NIC).
# Blocking sockets (used by requests/urllib3) work fine, so we implement the
# full WebSocket protocol over a plain ssl.wrap_socket() connection.

_FRAME_POLL_TIMEOUT = 5.0   # seconds between interrupt_flag checks


def _ws_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """Build a masked client→server WebSocket frame."""
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    plen   = len(payload)
    if plen < 126:
        hdr = bytes([0x80 | opcode, 0x80 | plen])
    elif plen < 65536:
        hdr = bytes([0x80 | opcode, 0x80 | 126]) + struct.pack(">H", plen)
    else:
        hdr = bytes([0x80 | opcode, 0x80 | 127]) + struct.pack(">Q", plen)
    return hdr + mask + masked


class _WsConn:
    """Blocking WebSocket connection used by both `_exec_sync` and
    `_ws_keepalive_loop`.  Holds the SSL socket plus any leftover bytes from
    the upgrade response, and exposes recv_frame / send / close.  No
    auto-reconnect — callers own the high-level loop."""

    def __init__(self, ssock: ssl.SSLSocket, leftover: bytes = b"") -> None:
        self.ssock = ssock
        self._buf  = bytearray(leftover)

    def settimeout(self, t: float | None) -> None:
        self.ssock.settimeout(t)

    def _read(self, n: int) -> bytes:
        while len(self._buf) < n:
            chunk = self.ssock.recv(max(n, 4096))
            if not chunk:
                raise EOFError("WebSocket connection closed")
            self._buf.extend(chunk)
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def recv_frame(self) -> tuple[int, bytes]:
        parts: list[bytes] = []
        final_op = 0x1
        while True:
            two    = self._read(2)
            b0, b1 = two[0], two[1]
            fin    = bool(b0 & 0x80)
            opcode = b0 & 0x0f
            masked = bool(b1 & 0x80)
            plen   = b1 & 0x7f
            if opcode:
                final_op = opcode
            if plen == 126:
                plen = struct.unpack(">H", self._read(2))[0]
            elif plen == 127:
                plen = struct.unpack(">Q", self._read(8))[0]
            mask_key = self._read(4) if masked else b""
            data     = self._read(plen)
            if masked:
                data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
            if final_op == 0x8:
                return 0x8, b""
            if final_op == 0x9:
                return 0x9, data
            parts.append(data)
            if fin:
                return final_op, b"".join(parts)

    def send(self, payload: bytes, opcode: int = 0x1) -> None:
        self.ssock.sendall(_ws_frame(payload, opcode))

    def close(self) -> None:
        try:
            self.ssock.close()
        except Exception:
            pass


def _ws_handshake(
    jupyter_url: str,
    proxy_token: str,
    auth_header: str,
    kernel_id: str,
    session_id: str,
    *,
    timeout: float = 60,
) -> _WsConn:
    """Open a Jupyter WebSocket to ``kernel_id`` and return a `_WsConn` with
    leftover post-upgrade bytes already buffered.  Raises on any failure."""
    p    = urllib.parse.urlparse(jupyter_url)
    host = p.hostname
    port = 443
    path = f"/api/kernels/{kernel_id}/channels?session_id={session_id}"

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.set_alpn_protocols(["http/1.1"])

    raw   = socket.create_connection((host, port), timeout=timeout)
    ssock = ssl_ctx.wrap_socket(raw, server_hostname=host)

    key  = base64.b64encode(os.urandom(16)).decode()
    hdrs = [
        f"GET {path} HTTP/1.1",
        f"Host: {host}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        f"Authorization: {auth_header}",
        f"X-Colab-Runtime-Proxy-Token: {proxy_token}",
        "X-Colab-Tunnel: Google",
        f"Origin: {jupyter_url}",
    ]
    ssock.sendall(("\r\n".join(hdrs) + "\r\n\r\n").encode())

    buf = b""
    ssock.settimeout(30)
    while b"\r\n\r\n" not in buf:
        chunk = ssock.recv(4096)
        if not chunk:
            try: ssock.close()
            except Exception: pass
            raise RuntimeError("Connection closed during WebSocket handshake")
        buf += chunk

    if b"101" not in buf.split(b"\r\n", 1)[0]:
        try: ssock.close()
        except Exception: pass
        raise RuntimeError(f"WS upgrade failed: {buf[:300]!r}")

    leftover = buf[buf.index(b"\r\n\r\n") + 4:]
    return _WsConn(ssock, leftover)


def _exec_sync(
    jupyter_url: str,
    proxy_token: str,
    auth_header: str,
    kernel_id: str,
    code: str,
    job: "_Job",
    interrupt_flag: threading.Event,
    timeout: float = 3600,
    show: bool = False,
    prelude: str = "",
) -> None:
    """
    Execute code on the Jupyter kernel via a synchronous blocking WebSocket.

    Uses raw ssl.wrap_socket() (same network path as requests) to avoid
    asyncio/Tailscale incompatibility.  Runs in a worker thread.

    ``prelude`` is prepended to ``code`` *only on the wire* — it does not
    affect anything passed in by the caller, so secret-injecting preludes
    don't leak into ``index.json``.
    """

    session_id = str(uuid.uuid4())
    msg_id     = str(uuid.uuid4())
    wire_code  = (prelude + code) if prelude else code

    execute_msg = {
        "header": {
            "msg_id":   msg_id,
            "msg_type": "execute_request",
            "username": "direct_kernel",
            "session":  session_id,
            "date":     "",
            "version":  "5.3",
        },
        "parent_header": {},
        "metadata":      {},
        "content": {
            "code":             wire_code,
            "silent":           False,
            "store_history":    True,
            "user_expressions": {},
            "allow_stdin":      False,
            "stop_on_error":    True,
        },
        "channel": "shell",
        "buffers": [],
    }

    def _emit(ev: dict) -> None:
        job._append(ev)
        if show:
            text = ev.get("text", "")
            if ev.get("type") in ("stderr", "error"):
                sys.stderr.write(text); sys.stderr.flush()
            else:
                sys.stdout.write(text); sys.stdout.flush()

    conn: _WsConn | None = None
    try:
        with _spin("connecting to kernel"):
            conn = _ws_handshake(jupyter_url, proxy_token, auth_header,
                                 kernel_id, session_id, timeout=60)

        conn.send(json.dumps(execute_msg).encode())

        # ── read frames until kernel goes idle ─────────────────────────────────
        # Show a spinner when no output has arrived recently so the user
        # knows something is happening during silent computations.
        conn.settimeout(_FRAME_POLL_TIMEOUT)
        deadline      = time.time() + timeout
        idle_seen     = False
        last_output   = time.time()
        exec_spin_stop: threading.Event | None = None

        def _maybe_start_exec_spin() -> None:
            nonlocal exec_spin_stop
            if exec_spin_stop is None and sys.stderr.isatty():
                exec_spin_stop = threading.Event()
                threading.Thread(
                    target=_spin_thread,
                    args=("executing", exec_spin_stop),
                    daemon=True,
                ).start()

        def _stop_exec_spin() -> None:
            nonlocal exec_spin_stop
            if exec_spin_stop is not None:
                exec_spin_stop.set()
                exec_spin_stop = None
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()

        while time.time() < deadline and not idle_seen:
            if interrupt_flag.is_set():
                break
            # Start spinner if no output for 1 second
            if time.time() - last_output > 1.0:
                _maybe_start_exec_spin()
            try:
                opcode, payload = conn.recv_frame()
            except (TimeoutError, socket.timeout):
                continue
            except EOFError:
                break

            if opcode == 0x8:   # close
                break
            if opcode == 0x9:   # ping → pong
                conn.send(payload, opcode=0xA)
                continue

            try:
                msg = json.loads(payload)
            except Exception:
                continue

            pid     = msg.get("parent_header", {}).get("msg_id", "")
            mtype   = msg.get("msg_type", "")
            content = msg.get("content", {})

            if mtype == "stream" and pid == msg_id:
                _stop_exec_spin()
                last_output = time.time()
                _emit({"type": content.get("name", "stdout"), "text": content.get("text", "")})

            elif mtype in ("display_data", "execute_result") and pid == msg_id:
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    _stop_exec_spin()
                    last_output = time.time()
                    _emit({"type": "result", "text": text + "\n", "data": content.get("data", {})})

            elif mtype == "error" and pid == msg_id:
                _stop_exec_spin()
                last_output = time.time()
                tb = "\n".join(content.get("traceback", []))
                _emit({"type": "error", "text": tb + "\n",
                       "ename": content.get("ename", ""), "evalue": content.get("evalue", "")})

            elif mtype == "status" and content.get("execution_state") == "idle" and pid == msg_id:
                idle_seen = True

        _stop_exec_spin()
        if not idle_seen and not interrupt_flag.is_set():
            _emit({"type": "timeout", "text": f"[direct_kernel] timed out after {timeout}s\n"})

    except Exception as exc:
        _emit({"type": "error", "text": f"[direct_kernel] exec error: {exc}\n{traceback.format_exc()}\n"})
    finally:
        if conn:
            conn.close()


# ── kernel-connections probe (diagnostic) ─────────────────────────────────────
#
# Periodically polls /api/kernels and logs the `connections` count for each
# kernel.  Used to verify (against `ws_coverage.log`) that our persistent WS
# keepalive is keeping connections >= 1 across job boundaries — i.e. that we
# are actually winning Colab's `kernels.list[].connections > 0` check.

_WS_COVERAGE_LOG = _JOBS_DIR / "ws_coverage.log"


def _ws_coverage_log(endpoint: str, msg: str) -> None:
    try:
        _JOBS_DIR.mkdir(exist_ok=True)
        with _WS_COVERAGE_LOG.open("a") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}  ep={endpoint[:38]:<38}  {msg}\n")
    except Exception:
        pass


_GRPC_KEEPALIVE_URL = ("https://colab.clients6.google.com/$rpc/"
                      "google.internal.colab.v1.RuntimeService/KeepAliveAssignment")
_GRPC_KEEPALIVE_API_KEY = "AIzaSyA2BvntLwNwFthUB4w6_Bhn0cMlVHwyaHc"


def _grpc_keepalive_ping(sess: requests.Session, endpoint: str) -> int:
    """POST to the same gRPC RPC the official client uses to keep a runtime
    alive: `RuntimeService.KeepAliveAssignment`.  Body is JSON-protobuf
    `[endpoint]`.  Returns HTTP status code.

    Captured from a real browser-Colab session via mitmproxy: the browser
    fires this every 60 s.  The bridge's pre-existing
    `/tun/m/{ep}/keep-alive/` ping is a *different* (older?) endpoint;
    Pro+ A100 preemption at ~20 min suggests Colab's scheduler watches
    THIS RPC, not the tun-based one.
    """
    hdrs = {
        "content-type": "application/json+protobuf",
        "origin":  "https://colab.research.google.com",
        "referer": "https://colab.research.google.com/",
        "x-goog-api-key":   _GRPC_KEEPALIVE_API_KEY,
        "x-goog-authuser":  "0",
        "x-user-agent":     "grpc-web-javascript/0.1",
    }
    r = sess.post(_GRPC_KEEPALIVE_URL, headers=hdrs, json=[endpoint], timeout=10)
    return r.status_code


def _grpc_keepalive_loop(endpoint: str, stop_event: threading.Event) -> None:
    """Send the gRPC keep-alive RPC every 60 s for the lifetime of the
    reaper.  Diagnostic logs: `colab_bridge/.direct_kernel_jobs/ws_coverage.log`."""
    sess = make_session()
    n = 0
    while not stop_event.is_set():
        try:
            _ensure_fresh_token(sess)
        except Exception:
            pass
        try:
            code = _grpc_keepalive_ping(sess, endpoint)
            n += 1
            if n <= 3 or n % 5 == 0 or code != 200:
                _ws_coverage_log(endpoint, f"grpc-keepalive #{n}: HTTP {code}")
        except Exception as exc:
            _ws_coverage_log(endpoint,
                             f"grpc-keepalive: {type(exc).__name__}: {exc}")
        if stop_event.wait(60):
            return


def _resources_poll_loop(endpoint: str, stop_event: threading.Event) -> None:
    """Mirror the VS Code Colab extension's `resource_poll_interval_ms` (~10s)
    polling of `GET /api/colab/resources`.  This is the only runtime-scoped
    recurring HTTP call from the official client that the bridge does not
    replicate; it's the prime candidate for the missing client signal Colab
    uses when deciding to preempt Pro+ A100s under capacity pressure."""
    sess = make_session()
    n = 0
    while not stop_event.is_set():
        try:
            fresh = _refresh_proxy_for_endpoint(sess, endpoint)
        except Exception:
            if stop_event.wait(10):
                return
            continue
        if fresh is None:
            if stop_event.wait(10):
                return
            continue
        jurl, ptok = fresh
        try:
            _ensure_fresh_token(sess)
        except Exception:
            pass
        try:
            r = sess.get(f"{jurl}/api/colab/resources",
                         headers={"X-Colab-Runtime-Proxy-Token": ptok,
                                  "X-Colab-Tunnel": "Google"},
                         timeout=10)
            n += 1
            if n <= 3 or n % 30 == 0 or r.status_code != 200:
                _ws_coverage_log(endpoint,
                                 f"resources-poll #{n}: HTTP {r.status_code} "
                                 f"({len(r.content)}B)")
        except Exception as exc:
            _ws_coverage_log(endpoint,
                             f"resources-poll: {type(exc).__name__}: {exc}")
        if stop_event.wait(10):
            return


def _connections_probe_loop(endpoint: str, stop_event: threading.Event) -> None:
    """Every 30 s, hit /api/kernels for the runtime and log
    `[{id, execution_state, connections}, …]`.  Read-only — does not affect
    any state.  Output: colab_bridge/.direct_kernel_jobs/ws_coverage.log."""
    sess = make_session()
    while not stop_event.is_set():
        try:
            fresh = _refresh_proxy_for_endpoint(sess, endpoint)
        except Exception as exc:
            _ws_coverage_log(endpoint, f"probe: list_runtimes raised {type(exc).__name__}")
            if stop_event.wait(30):
                return
            continue
        if fresh is None:
            _ws_coverage_log(endpoint, "probe: endpoint missing from list_runtimes")
            if stop_event.wait(30):
                return
            continue
        jurl, ptok = fresh
        try:
            _ensure_fresh_token(sess)
        except Exception:
            pass
        try:
            r = sess.get(f"{jurl}/api/kernels",
                         headers={"X-Colab-Runtime-Proxy-Token": ptok,
                                  "X-Colab-Tunnel": "Google"},
                         timeout=10)
            if r.status_code == 200:
                kernels = r.json()
                summary = ", ".join(
                    f"{k.get('id','?')[:8]}:{k.get('execution_state','?')}"
                    f":conn={k.get('connections','?')}" for k in kernels)
                _ws_coverage_log(endpoint, f"probe: {summary or '(no kernels)'}")
            else:
                _ws_coverage_log(endpoint, f"probe: HTTP {r.status_code}")
        except Exception as exc:
            _ws_coverage_log(endpoint, f"probe: request failed {type(exc).__name__}: {exc}")
        if stop_event.wait(30):
            return


# ── persistent Jupyter-WS keepalive ───────────────────────────────────────────
#
# Colab's HTTP /tun/m/{ep}/keep-alive/ ping is conditional on
# kernels.list[].connections > 0 (see VS Code extension's `keepServerAlive`).
# Without an open Jupyter WS, the ping is suppressed and the scheduler
# reclaims our runtime as idle.  This loop holds one WS open per runtime for
# the reaper's lifetime.

_WS_KA_BACKOFF = (1, 2, 5, 15, 30, 60)


def _ws_keepalive_loop(
    endpoint: str,
    kernel_id_hint: str | None,
    stop_event: threading.Event,
) -> None:
    """Persistent Jupyter WebSocket so Colab's idle scheduler sees this
    runtime's ``connections`` count stay >= 1.  Runs as a daemon thread inside
    `_internal_reap`; exits cleanly on ``stop_event``, on observed
    ``released_at``, or after a confirmed-preempted double-check."""
    sess       = make_session()
    session_id = str(uuid.uuid4())
    bidx           = 0
    consec_missing = 0
    first_miss_at  = 0.0
    cached_kid     = kernel_id_hint

    def _back(reason: str) -> bool:
        nonlocal bidx
        wait = _WS_KA_BACKOFF[min(bidx, len(_WS_KA_BACKOFF) - 1)]
        bidx += 1
        return stop_event.wait(wait)

    while not stop_event.is_set():
        if (_runtime_meta(endpoint) or {}).get("released_at"):
            _reap_log(endpoint, "ws-keepalive: released_at observed, exiting")
            return

        # Preemption probe: list_runtimes() miss is the ONLY signal that
        # triggers stamping released_by="preempted" — and only after the
        # double-check (>=2 consecutive misses, ≥60s apart).  All other
        # failures (handshake 4xx, EOF, socket errors, network flakes) are
        # silent reconnects.  This conservative rule prevents the
        # false-positive preemption bug seen in past versions.
        try:
            fresh = _refresh_proxy_for_endpoint(sess, endpoint)
        except Exception as exc:
            _reap_log(endpoint,
                      f"ws-keepalive: list_runtimes raised {type(exc).__name__}, ignoring")
            consec_missing = 0
            first_miss_at  = 0.0
            if _back("api-error"):
                return
            continue

        if fresh is None:
            consec_missing += 1
            if first_miss_at == 0.0:
                first_miss_at = time.monotonic()
            elapsed = time.monotonic() - first_miss_at
            if consec_missing >= 2 and elapsed > 60:
                _reap_log(endpoint,
                          f"ws-keepalive: PREEMPTED — endpoint missing "
                          f"{consec_missing}x over {elapsed:.0f}s")
                try:
                    _set_runtime_meta(endpoint,
                                      released_at=time.time(),
                                      released_by="preempted",
                                      preempted=True)
                    _emit_event("runtime_released", endpoint=endpoint, reason="preempted")
                except Exception:
                    pass
                return
            _reap_log(endpoint,
                      f"ws-keepalive: endpoint missing (consec={consec_missing}, "
                      f"elapsed={elapsed:.0f}s) — not yet preempted")
            if _back("missing"):
                return
            continue

        consec_missing = 0
        first_miss_at  = 0.0
        jurl, ptok     = fresh

        try:
            _ensure_fresh_token(sess)
        except Exception:
            pass
        auth = sess.headers.get("Authorization", "")

        if not cached_kid:
            try:
                cached_kid = get_or_create_kernel(sess, jurl, ptok)
            except Exception as exc:
                _reap_log(endpoint,
                          f"ws-keepalive: get_or_create_kernel failed: {type(exc).__name__}")
                if _back("kernel"):
                    return
                continue

        try:
            conn = _ws_handshake(jurl, ptok, auth, cached_kid, session_id, timeout=30)
        except Exception as exc:
            msg = str(exc)
            if "401" in msg or "403" in msg:
                try:
                    _ensure_fresh_token(sess)
                    auth = sess.headers.get("Authorization", "")
                    conn = _ws_handshake(jurl, ptok, auth, cached_kid, session_id, timeout=30)
                except Exception as exc2:
                    _reap_log(endpoint,
                              f"ws-keepalive: handshake auth-retry failed: "
                              f"{type(exc2).__name__}")
                    cached_kid = None    # kernel may have been recreated
                    if _back("auth"):
                        return
                    continue
            else:
                _reap_log(endpoint,
                          f"ws-keepalive: handshake failed: {type(exc).__name__}: {exc}")
                cached_kid = None
                if _back("handshake"):
                    return
                continue

        bidx = 0    # successful handshake → reset backoff
        _reap_log(endpoint, f"ws-keepalive: connected to kernel {cached_kid[:8]}")

        connected_at    = time.monotonic()
        last_hb         = 0.0
        last_meta_check = time.monotonic()

        try:
            conn.settimeout(5)
            while not stop_event.is_set():
                if time.monotonic() - connected_at > 3300:
                    _reap_log(endpoint, "ws-keepalive: rotating connection (token age)")
                    break

                if time.monotonic() - last_hb > 25:
                    hb = {
                        "header": {
                            "msg_id":   uuid.uuid4().hex,
                            "msg_type": "kernel_info_request",
                            "username": "direct_kernel_keepalive",
                            "session":  session_id,
                            "date":     "",
                            "version":  "5.3",
                        },
                        "parent_header": {},
                        "metadata":      {},
                        "content":       {},
                        "channel":       "shell",
                        "buffers":       [],
                    }
                    try:
                        conn.send(json.dumps(hb).encode())
                        last_hb = time.monotonic()
                    except Exception as exc:
                        _reap_log(endpoint,
                                  f"ws-keepalive: send failed ({type(exc).__name__}), reconnecting")
                        break

                if time.monotonic() - last_meta_check > 30:
                    if (_runtime_meta(endpoint) or {}).get("released_at"):
                        _reap_log(endpoint, "ws-keepalive: released_at observed in inner loop, exiting")
                        return
                    last_meta_check = time.monotonic()

                try:
                    opcode, payload = conn.recv_frame()
                except (TimeoutError, socket.timeout):
                    continue
                except EOFError:
                    _reap_log(endpoint, "ws-keepalive: EOF, reconnecting")
                    break
                except (OSError, ssl.SSLError) as exc:
                    _reap_log(endpoint,
                              f"ws-keepalive: socket error ({type(exc).__name__}), reconnecting")
                    break

                if opcode == 0x8:
                    _reap_log(endpoint, "ws-keepalive: server CLOSE, reconnecting")
                    break
                if opcode == 0x9:
                    try:
                        conn.send(payload, opcode=0xA)
                    except Exception:
                        break
                    continue
                # Data frames (kernel_info_reply, status idle, etc.) — discard.
        finally:
            conn.close()

        if stop_event.is_set():
            break
        if _back("reconnect"):
            return

    _reap_log(endpoint, "ws-keepalive: stop_event set, exiting")


# ── ColabDirectKernel ──────────────────────────────────────────────────────────

class ColabDirectKernel:
    """
    High-level interface: OAuth auth, runtime assignment, job queuing, streaming,
    kernel restart, graceful + force interrupt.

    Quick start:
        dk = ColabDirectKernel.connect(accelerator="A100")
        dk.start_keepalive()

        # Blocking:
        dk.run("import torch; print(torch.cuda.get_device_name(0))")

        # Non-blocking:
        jid = dk.submit("long_computation()")
        for ev in dk.stream(jid):
            print(ev)

        # Interrupt a frozen job:
        dk.interrupt()        # graceful SIGINT
        dk.force_interrupt()  # kill + recreate kernel

        # Release runtime:
        dk.unassign()
    """

    def __init__(
        self,
        sess: requests.Session,
        jupyter_url: str,
        proxy_token: str,
        kernel_id: str,
        endpoint: str,
        accelerator: str = "?",
        store: _JobStore | None = None,
    ) -> None:
        self.sess        = sess
        self.jupyter_url = jupyter_url
        self.proxy_token = proxy_token
        self.kernel_id   = kernel_id
        self.endpoint    = endpoint
        self.accelerator = accelerator
        self.store       = store if store is not None else _JobStore()

        self._jobs: dict[str, _Job]     = {}
        self._queue: Queue[_Job | None] = Queue()
        self._current: _Job | None      = None
        self._interrupt                 = threading.Event()
        self._lock                      = threading.Lock()

        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="dk-worker"
        )
        self._worker_thread.start()
        self._ka_stop: threading.Event | None = None

    # ── factories ──────────────────────────────────────────────────────────────

    @classmethod
    def connect(
        cls,
        accelerator: str = "A100",
        high_ram: bool = False,
        file_id: str | None = None,
    ) -> "ColabDirectKernel":
        """OAuth → notebook → assign → kernel.  Returns ready instance."""
        sess            = make_session()
        fid             = file_id or get_or_create_notebook(sess)
        jurl, ptok, ep  = assign_runtime(sess, fid, accelerator, high_ram)
        kid             = get_or_create_kernel(sess, jurl, ptok)
        return cls(sess, jurl, ptok, kid, ep, accelerator=accelerator.upper())

    # ── background worker ──────────────────────────────────────────────────────

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                break
            with self._lock:
                self._current = job
            self._interrupt.clear()
            job.status = "running"
            self.store.update_job(
                job.job_id,
                status="running",
                kernel_id=self.kernel_id,    # may have changed since submit
                started=time.time(),
            )
            _write_job_pid(job.job_id)
            _emit_event("job_started", jid=job.job_id,
                        endpoint=self.endpoint, accel=self.accelerator)

            # Refresh token before long jobs
            _ensure_fresh_token(self.sess)
            auth = self.sess.headers.get("Authorization", "")

            # Build env-injection prelude from the choice we recorded at submit.
            rec     = self.store.get_job(job.job_id) or {}
            inject  = rec.get("inject_env", True)
            prelude = _build_env_prelude(_resolve_env_for_job(inject))

            try:
                _exec_sync(
                    self.jupyter_url, self.proxy_token, auth,
                    self.kernel_id, job.code, job, self._interrupt,
                    prelude=prelude,
                )
                has_error = any(e["type"] == "error" for e in job.events)
                job._finish("error" if has_error else "done")
            except Exception as exc:
                job._append({"type": "error", "text": f"[worker] {exc}\n"})
                job._finish("error")
            finally:
                _clear_job_pid(job.job_id)

            with self._lock:
                self._current = None
            self._queue.task_done()

    # ── job API ────────────────────────────────────────────────────────────────

    def submit(self, code: str, *, inject_env: bool | list[str] = True,
               desc: str | None = None) -> str:
        """Queue code for execution.

        ``inject_env``:
          * ``True``  (default) — inject every key in ``colab_bridge/.env`` as
            ``os.environ`` before the user's code runs.
          * ``False`` — skip env injection.
          * list of names — inject only those names from ``.env``.

        ``desc``: optional one-line label for the job, surfaced in
        ``--jobs`` / ``--follow`` / ``--live``.

        Returns the job_id immediately (non-blocking).
        """
        jid = str(uuid.uuid4())[:8]
        job = _Job(jid, code, store=self.store)
        self._jobs[jid] = job
        self.store.add_job({
            "job_id":      jid,
            "status":      "queued",
            "endpoint":    self.endpoint,
            "accelerator": self.accelerator,
            "jupyter_url": self.jupyter_url,
            "kernel_id":   self.kernel_id,
            "code":        code,
            "desc":        desc,
            "inject_env":  inject_env,
            "started":     time.time(),
            "ended":       None,
        })
        self._queue.put(job)
        print(f"[direct_kernel] queued job {jid} (runtime: {self.endpoint})", flush=True)
        return jid

    def stream(
        self,
        job_id: str,
        *,
        show: bool = True,
        since: int = 0,
    ) -> Iterator[dict]:
        """Yield events from a job as they arrive; blocks until done."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job_id: {job_id!r}")
        for ev in job.iter_events(since=since):
            if show:
                text = ev.get("text", "")
                if ev.get("type") in ("stderr", "error"):
                    sys.stderr.write(text); sys.stderr.flush()
                else:
                    sys.stdout.write(text); sys.stdout.flush()
            yield ev

    def run(self, code: str, *, show: bool = True,
            inject_env: bool | list[str] = True,
            desc: str | None = None) -> list[dict]:
        """Submit and block until complete, streaming output.  Returns events."""
        return list(self.stream(
            self.submit(code, inject_env=inject_env, desc=desc), show=show))

    def run_file(self, path: str | Path, *, show: bool = True,
                 inject_env: bool | list[str] = True,
                 desc: str | None = None) -> list[dict]:
        """Read a .py file and execute it on the Colab runtime."""
        return self.run(Path(path).read_text(), show=show,
                        inject_env=inject_env, desc=desc)

    def jobs(self) -> list[dict]:
        """Return a status snapshot of all submitted jobs."""
        return [j.summary() for j in self._jobs.values()]

    def job_events(self, job_id: str) -> list[dict]:
        """Return all stored events for a completed (or in-progress) job."""
        return list(self._jobs[job_id].events)

    # ── interrupt / force-interrupt ────────────────────────────────────────────

    def interrupt(self) -> bool:
        """
        Graceful interrupt: send SIGINT to the running kernel.
        Returns True if the HTTP request succeeded.
        """
        self._interrupt.set()
        ok = _do_interrupt(self.sess, self.jupyter_url, self.proxy_token, self.kernel_id)
        with self._lock:
            job = self._current
        if job and job.status == "running":
            job._append({"type": "stderr", "text": "\n[direct_kernel] KeyboardInterrupt\n"})
            job._finish("cancelled")
        print("[direct_kernel] Interrupt sent.", flush=True)
        return ok

    def force_interrupt(self) -> None:
        """
        Force-interrupt: kill the kernel and immediately recreate it.
        Use when the kernel is frozen and ignores graceful interrupt.
        """
        self._interrupt.set()
        with self._lock:
            job = self._current
        if job and job.status == "running":
            job._append({"type": "stderr", "text": "\n[direct_kernel] FORCE INTERRUPT — kernel killed\n"})
            job._finish("cancelled")
        self.kernel_id = _do_force_restart(
            self.sess, self.jupyter_url, self.proxy_token, self.kernel_id
        )
        self._interrupt.clear()
        print("[direct_kernel] Force interrupt done; new kernel ready.", flush=True)

    # ── kernel restart ─────────────────────────────────────────────────────────

    def restart_kernel(self) -> None:
        """Restart the kernel (clears variables).  Cancels any running job."""
        self._interrupt.set()
        with self._lock:
            job = self._current
        if job and job.status == "running":
            job._finish("cancelled")
        self.kernel_id = _do_restart(
            self.sess, self.jupyter_url, self.proxy_token, self.kernel_id
        )
        self._interrupt.clear()

    # ── keep-alive ─────────────────────────────────────────────────────────────

    def start_keepalive(self, interval: float = 60.0) -> None:
        """Start a background daemon thread that pings the runtime keep-alive."""
        self._ka_stop = threading.Event()
        threading.Thread(
            target=_keepalive_loop,
            args=(self.sess, self.endpoint, self._ka_stop),
            kwargs={"interval": interval},
            daemon=True, name="dk-keepalive",
        ).start()

    def stop_keepalive(self) -> None:
        if self._ka_stop:
            self._ka_stop.set()

    # ── runtime lifecycle ──────────────────────────────────────────────────────

    def unassign(self) -> None:
        """Release this runtime back to the Colab pool."""
        unassign_runtime(self.sess, self.endpoint)
        print(f"[direct_kernel] Released {self.endpoint}.", flush=True)

    def close(self) -> None:
        """Stop worker thread; call before exiting."""
        self._queue.put(None)
        self._worker_thread.join(timeout=5)

    def __repr__(self) -> str:
        return (f"ColabDirectKernel(endpoint={self.endpoint!r}, "
                f"kernel={self.kernel_id!r}, jobs={len(self._jobs)})")

    def __enter__(self) -> "ColabDirectKernel":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _fmt_hours(h: float) -> str:
    """Pretty-format an hour count: minutes for <1h, hours for <1d, else days."""
    if h < 0:
        return "?"
    if h < 1:
        return f"{h*60:.1f}m"
    if h < 24:
        return f"{h:.2f}h"
    return f"{h/24:.1f}d"


def _trunc(s: str | None, n: int = 25) -> str:
    """Trim string to n chars with a … suffix on overflow.  None / empty → ''."""
    if not s:
        return ""
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_age(t: float | None) -> str:
    if not t:
        return "—"
    dt = time.time() - t
    if dt < 60:    return f"{dt:.0f}s ago"
    if dt < 3600:  return f"{dt/60:.0f}m ago"
    if dt < 86400: return f"{dt/3600:.1f}h ago"
    return f"{dt/86400:.1f}d ago"


def _fmt_dur(rec: dict) -> str:
    s = rec.get("started"); e = rec.get("ended") or time.time()
    if not s: return "—"
    d = e - s
    if d < 60:    return f"{d:.1f}s"
    if d < 3600:  return f"{d/60:.1f}m"
    return f"{d/3600:.1f}h"


def _print_jobs_table(jobs: list[dict]) -> None:
    if not jobs:
        print("No jobs recorded.")
        return
    # Oldest first → newest at the bottom, right above your prompt.
    jobs = sorted(jobs, key=lambda r: r.get("started", 0))
    # Build SH map from currently-active runtimes (released ones don't have a letter).
    try:
        active_rts = list_runtimes(make_session())
    except Exception:
        active_rts = []
    sh_map = _shorthand_map(active_rts)
    ep_to_letter = {ep: sh for sh, ep in sh_map.items()}
    print(f"{'JOB_ID':<10} {'STATUS':<10} {'SH':<3} {'RUNTIME':<38} {'ACCEL':<6} {'STARTED':<14} {'ELAPSED':<8} {'DESC':<26} CODE")
    print("─" * 150)
    for r in jobs:
        code = (r.get("code") or "").replace("\n", " ⏎ ")[:35]
        sh   = ep_to_letter.get(r.get("endpoint", ""), "—")
        desc = _trunc(r.get("desc"), 25) or "—"
        # The store flips status="running" the instant the watcher attaches,
        # but the kernel may still be busy with a prior cell.  Surface that
        # as "queued" everywhere we display status.  Same logic as --live.
        conn = _job_connection_status(r)
        status_disp = "queued" if conn == "queued" else r["status"]
        print(
            f"{r['job_id']:<10} {status_disp:<10} {sh:<3} {r.get('endpoint','—')[:38]:<38} "
            f"{r.get('accelerator','—'):<6} {_fmt_age(r.get('started')):<14} {_fmt_dur(r):<8} {desc:<26} {code}"
        )


def _parse_region_from_url(url: str) -> str:
    """Extract the GCP region from a Colab runtime proxy URL.

    URLs have the form
      https://<port>-<endpoint>-<c|b>.<region>-<index>.prod.colab.dev
    e.g. ``https://8080-m-s-kkb-usc1c1-...-c.us-central1-1.prod.colab.dev`` →
    ``us-central1``.
    """
    try:
        host = url.split("//", 1)[1].split("/", 1)[0]
        if not host.endswith(".prod.colab.dev"):
            return "?"
        rest = host[: -len(".prod.colab.dev")]
        last = rest.rsplit(".", 1)[-1]      # "us-central1-1"
        return last.rsplit("-", 1)[0]       # "us-central1"
    except Exception:
        return "?"


def _print_runtimes_table(rts: list[dict]) -> None:
    if not rts:
        print("No active runtimes.")
        return
    # Letters apply only to currently-active runtimes (released ones don't get one).
    active_only = [r for r in rts if not r.get("_released_at")]
    sh_map = _shorthand_map(active_only)
    inv    = {ep: sh for sh, ep in sh_map.items()}
    # Order: oldest first, newest at the bottom.  Released ones go ABOVE active
    # ones (oldest release at top, newest active at bottom = right by your prompt).
    released_sorted = sorted(
        [r for r in rts if r.get("_released_at")],
        key=lambda r: r.get("_released_at", 0),
    )
    active_sorted = sorted(
        active_only,
        key=lambda r: (_runtime_meta(r["endpoint"]) or {}).get("assigned_at", 0),
    )
    rts = released_sorted + active_sorted

    has_released = any(r.get("_released_at") for r in rts)
    if has_released:
        print(f"{'SH':<3} {'STATUS':<9} {'ENDPOINT':<38} {'ACCEL':<6} {'REGION':<16} {'TIMEOUT':<8} {'AGE':<14} {'DESC':<26}")
        print("─" * 126)
    else:
        print(f"{'SH':<3} {'ENDPOINT':<38} {'ACCEL':<6} {'SHAPE':<10} {'REGION':<16} {'TIMEOUT':<10} {'DESC':<26}")
        print("─" * 116)
    now = time.time()
    for r in rts:
        accel  = _normalized_accel(r)
        ep     = r.get("endpoint", "?")
        meta   = _runtime_meta(ep) or {}
        # Trust the live-API record over local meta: only treat the explicit
        # `_released_at` (set by our CLI handler when augmenting from history)
        # as the released signal.  Don't fall back to meta — a runtime that's
        # in the live API list is by definition active, even if local meta
        # has a stale `released_at` from a previous lifecycle / racy stamp.
        released_at = r.get("_released_at")
        if released_at:
            t = meta.get("idle_timeout_min")
            t_str  = "—" if t is None else ("off" if t == 0 else f"{t}m")
            assigned = meta.get("assigned_at")
            ran   = (released_at - assigned) / 3600 if assigned else None
            age_h = (now - released_at) / 3600
            age_s = f"{_fmt_hours(age_h)} ago"
            ran_s = f" (ran {_fmt_hours(ran)})" if ran is not None else ""
            region = meta.get("region") or "—"
            desc = _trunc(meta.get("desc"), 25) or "—"
            print(f"{'─':<3} {'released':<9} {ep[:38]:<38} {accel:<6} {region:<16} {t_str:<8} {age_s+ran_s:<22} {desc:<26}")
            continue

        # Active runtime
        shape  = (r.get("machineShape", "?") or "?").replace("SHAPE_", "")
        region = _parse_region_from_url(r.get("runtimeProxyInfo", {}).get("url", ""))
        sh     = inv.get(ep, "?")
        t      = meta.get("idle_timeout_min")
        t_str  = "—" if t is None else ("off" if t == 0 else f"{t}m")
        desc   = _trunc(meta.get("desc"), 25) or "—"
        if has_released:
            assigned = meta.get("assigned_at")
            age = ((now - assigned) / 3600) if assigned else None
            age_s = _fmt_hours(age) if age is not None else "?"
            print(f"{sh:<3} {'active':<9} {ep[:38]:<38} {accel:<6} {region:<16} {t_str:<8} {age_s:<22} {desc:<26}")
        else:
            print(f"{sh:<3} {ep[:38]:<38} {accel:<6} {shape:<10} {region:<16} {t_str:<10} {desc:<26}")


# ── per-job handler pidfiles ─────────────────────────────────────────────────
# When a process (foreground CLI, watcher subprocess, or worker thread's
# parent) starts handling a job, it writes its PID to <job_id>.pid.  When the
# job finishes via `_finish` the pidfile is removed.  Reconcile checks the
# pidfile to detect zombies whose handler died without updating status.

def _job_pidfile(job_id: str) -> Path:
    return _JOBS_DIR / f"{job_id}.pid"


def _write_job_pid(job_id: str, pid: int | None = None) -> None:
    try:
        _JOBS_DIR.mkdir(exist_ok=True)
        _job_pidfile(job_id).write_text(str(pid if pid is not None else os.getpid()))
    except Exception:
        pass


def _clear_job_pid(job_id: str) -> None:
    p = _job_pidfile(job_id)
    if p.exists():
        try: p.unlink()
        except Exception: pass


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _job_connection_status(rec: dict) -> str:
    """Categorize how well a running job's local handler is hearing the kernel.

    Returns one of:
      "connected"    — recent events from the watcher (mtime ≤ 60s).
      "stale"        — events 60s–5min old; cell may just be silent.
      "disconnected" — events >5min old, OR pidfile points at a dead PID.
                       The job is still running on Colab but no one's
                       capturing output locally; use `colab --reattach JID`
                       to recover.
      "queued"       — store says queued, OR `running` but no events yet
                       AND we're past the 30s warmup grace.  This second
                       case happens when the watcher has attached but the
                       kernel is still busy with a prior cell — the bridge
                       marks the job "running" the instant `_internal_watch`
                       sends the execute_request, even though Jupyter
                       queues it.
      "n/a"          — job is in a terminal status (done / error / cancelled).
    """
    status = rec.get("status")
    if status not in ("running", "queued"):
        return "n/a"
    if status == "queued":
        return "queued"
    jid = rec.get("job_id")
    if jid and _is_handler_dead(jid):
        return "disconnected"
    try:
        ev = _JobStore().events_path(jid) if jid else None
    except Exception:
        ev = None
    if not ev or not ev.exists():
        # Watcher attached but no events yet.  During the first 30s after
        # `started` this is normal warmup; after that it almost always
        # means the kernel is busy with a prior cell and our request is
        # queued behind it.  Show as "queued", not "stale".
        started = rec.get("started") or 0
        return "connected" if (time.time() - started) < 30 else "queued"
    age = time.time() - ev.stat().st_mtime
    if age <= 60:    return "connected"
    if age <= 300:   return "stale"
    return "disconnected"


def _is_handler_dead(job_id: str) -> bool:
    """True iff a pidfile exists AND it points to a dead PID.

    Returns False when the pidfile is missing — that's uncertain (e.g. a
    legacy job from before the pidfile feature, or any other reason) and we
    must NOT mark such jobs as orphaned, otherwise we'd false-positive every
    pre-existing job and the reaper would unassign live runtimes.
    """
    p = _job_pidfile(job_id)
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
    except Exception:
        return False
    return not _is_pid_alive(pid)


def _reconcile_orphan_jobs(sess: requests.Session | None = None) -> int:
    """Mark as ``error`` any queued/running jobs that can no longer make
    progress.  Two failure modes:

      1. **Endpoint gone**: the runtime got unassigned (or Colab killed it)
         while a job was running.  The endpoint isn't in ``list_runtimes``
         anymore.
      2. **Handler dead**: the runtime is alive but the foreground / watcher
         process that was driving the job has died (terminal closed,
         OS-killed, etc.).  Detected via pidfile.

    Returns the number of jobs reconciled.  Skips jobs younger than 30s.
    """
    store     = _JobStore()
    open_jobs = [j for j in store.list_jobs() if j.get("status") in ("queued", "running")]
    if not open_jobs:
        return 0
    try:
        live_eps = {r["endpoint"] for r in list_runtimes(sess or make_session())}
    except Exception:
        live_eps = None    # network failure — only do pidfile check
    now = time.time()
    n = 0
    for j in open_jobs:
        if now - (j.get("started") or 0) < 30:
            continue
        jid = j["job_id"]
        ep  = j.get("endpoint")

        if live_eps is not None and ep not in live_eps:
            store.update_job(jid, status="error", ended=now)
            store.append_event(jid, {
                "type": "error",
                "text": f"[direct_kernel] runtime {ep} no longer assigned; "
                        "watcher orphaned, marking job as error.\n",
            })
            _clear_job_pid(jid)
            n += 1
            continue

        if _is_handler_dead(jid):
            store.update_job(jid, status="error", ended=now)
            store.append_event(jid, {
                "type": "error",
                "text": "[direct_kernel] handler process is gone (watcher / "
                        "foreground died); job is unrecoverable.\n",
            })
            _clear_job_pid(jid)
            n += 1
    return n


def _filter_jobs_by_runtime(jobs: list[dict], query: str) -> list[dict]:
    """Resolve a runtime query against a list of historical jobs.

    Tried in order: ``all`` → accelerator name → letter shorthand of currently
    -active runtimes → endpoint substring.  Letters work even if the runtime
    is gone (we still want ``--jobs a`` to be intuitive after assignment), but
    will fall through to substring matching otherwise.
    """
    if query.lower() == "all":
        return jobs
    q = query.upper()

    accel_hits = [j for j in jobs if (j.get("accelerator") or "").upper() == q]
    if accel_hits:
        return accel_hits

    try:
        rts = list_runtimes(make_session())
        sh  = _shorthand_map(rts)
        if query in sh:
            return [j for j in jobs if j.get("endpoint") == sh[query]]
    except Exception:
        pass

    return [j for j in jobs if query in (j.get("endpoint") or "")]


def _refresh_proxy_for_endpoint(sess: requests.Session, endpoint: str) -> tuple[str, str] | None:
    """Look up a runtime by endpoint and return a fresh (jupyter_url, proxy_token)."""
    for r in list_runtimes(sess):
        if r.get("endpoint") == endpoint:
            info = r.get("runtimeProxyInfo", {})
            return info["url"].rstrip("/"), info["token"]
    return None


def _is_preempted_double_check(
    sess: requests.Session,
    endpoint: str,
    *,
    second_check_delay: float = 60.0,
) -> bool:
    """Return True only if `list_runtimes()` shows the endpoint missing on
    TWO consecutive checks ≥`second_check_delay` seconds apart.

    Used by every preemption-stamping path (legacy `_keepalive_loop`, the
    reaper's periodic cross-check, and the reaper's `_ping`-based branch)
    to prevent a single transient `list_runtimes` blip — empty list from a
    Colab API hiccup, network 5xx, eventually-consistent index — from being
    misread as a real preemption and stamping `released_by="preempted"` on
    a runtime that's actually still alive.

    The cost: when preemption IS real, we delay declaration by
    `second_check_delay` seconds.  Worth it for correctness.
    """
    try:
        live = {r["endpoint"] for r in list_runtimes(sess)}
    except Exception:
        return False    # API error → don't trust
    if endpoint in live:
        return False
    time.sleep(second_check_delay)
    try:
        live = {r["endpoint"] for r in list_runtimes(sess)}
    except Exception:
        return False    # second-check API error → still don't trust
    return endpoint not in live


def _resolve_endpoints(query: str, runtimes: list[dict]) -> list[str]:
    """Resolve a user-typed shorthand to one or more runtime endpoints.

    Resolution order (first non-empty match wins):

      1. ``"all"`` — every active runtime.
      2. Accelerator name (``A100``, ``T4``, ``CPU``, …) — every runtime of that type.
      3. Letter shorthand (``a``, ``b``, ``c``, …) — one runtime, ordered by endpoint.
      4. Exact endpoint — one runtime.
      5. Endpoint prefix — must be unique; errors otherwise.
      6. Endpoint substring — must be unique; errors otherwise.

    Always returns a non-empty list (since accelerator and ``all`` are allowed
    to match many).  Raises ``ValueError`` on no-match or ambiguous prefix /
    substring.
    """
    if not runtimes:
        raise ValueError("No active runtimes.")
    eps = [r["endpoint"] for r in runtimes]

    # 1. all
    if query.lower() == "all":
        return list(eps)

    # 2. accelerator (multi-match allowed)
    q = query.upper()
    accel_hits = [r["endpoint"] for r in runtimes if _normalized_accel(r).upper() == q]
    if accel_hits:
        return accel_hits

    # 3. letter shorthand
    sh = _shorthand_map(runtimes)
    if query in sh:
        return [sh[query]]

    # 4. exact endpoint
    if query in eps:
        return [query]

    # 5. unique prefix
    pref = [ep for ep in eps if ep.startswith(query)]
    if len(pref) == 1:
        return pref
    if len(pref) > 1:
        raise ValueError(f"{query!r} is ambiguous (prefix matches):\n  " +
                         "\n  ".join(pref))

    # 6. unique substring
    sub = [ep for ep in eps if query in ep]
    if len(sub) == 1:
        return sub
    if len(sub) > 1:
        raise ValueError(f"{query!r} is ambiguous (substring matches):\n  " +
                         "\n  ".join(sub))

    raise ValueError(f"No runtime matches {query!r}.\nActive runtimes:\n  " +
                     "\n  ".join(eps))


# ── --follow multiplexer ──────────────────────────────────────────────────────

# 26 ANSI fg colors — one per letter shorthand (a→[0], b→[1], …, z→[25]).
# Indices > 25 (aa, ab, …) wrap by using the trailing letter, so 'aa' = 'a' = [0]
# and same color shows up across re-letterings as long as the trailing letter
# is stable.  Reserved separately: 91 (red) for stderr/error, 36 (cyan) for
# system / lifecycle, 33 (yellow) for warnings.
_LETTER_COLOR_PALETTE = [
    32, 33, 34, 35, 92, 93, 94, 95, 96,    # base 9
    31, 92, 93, 94, 95, 96, 32, 33, 34,    # next 9 (some repeats — fine)
    35, 92, 93, 94, 95, 96, 32, 33,         # final 8 → 26 total
]
assert len(_LETTER_COLOR_PALETTE) >= 26


def _color_for_letter(sh: str) -> int:
    """Deterministic color for a runtime shorthand letter.

    Index by the LAST char of the letter (so 'a' / 'aa' / 'ba' all share a
    color — same trailing letter means same color).  This is intentional: the
    common case is single-letter shorthands.
    """
    if not sh:
        return 36
    ch = sh[-1].lower()
    if "a" <= ch <= "z":
        return _LETTER_COLOR_PALETTE[ord(ch) - ord("a")]
    return 36


def _follow_all(filter_endpoints: set[str] | None = None) -> None:
    """Live multiplexer: tail every running job + watch runtime/job lifecycle.

    If ``filter_endpoints`` is given, only show events from jobs on those
    runtimes (and only their assign/unassign notifications).
    """
    store = _JobStore()
    sess  = make_session()

    # Mark any zombie jobs (status=running but their runtime is gone) as error
    # so they don't show up in the "in progress" seed below.
    _reconcile_orphan_jobs(sess)

    # Make sure every active runtime has a stable letter stamped.
    try:
        _stamp_missing_letters(list_runtimes(sess))
    except Exception:
        pass

    print_lock     = threading.Lock()
    stop           = threading.Event()
    seen_jobs:     dict[str, str]  = {}      # jid → status
    seen_rts:      dict[str, dict] = {}      # endpoint → record
    started:       set[str]        = set()   # jids whose streamer is already running

    def emit(prefix: str, msg: str, color: int | None = None,
             text_color: int | None = None) -> None:
        """Emit one prefixed line.  ``color`` colors the prefix.  Optional
        ``text_color`` lets the message body have a different color than the
        prefix — used for stderr (prefix in the runtime's letter color so
        you can still tell which job, body in red so you can tell it's an
        error)."""
        with print_lock:
            pfx = f"\033[{color}m{prefix}\033[0m" if color else prefix
            txt = f"\033[{text_color}m{msg}\033[0m" if text_color else msg
            sys.stdout.write(f"{pfx} {txt}\n")
            sys.stdout.flush()

    def emit_block(lines: list[str], color: int | None = None) -> None:
        with print_lock:
            for ln in lines:
                if color:
                    sys.stdout.write(f"\033[{color}m{ln}\033[0m\n")
                else:
                    sys.stdout.write(ln + "\n")
            sys.stdout.flush()

    def sh_for(endpoint: str) -> str:
        sh_map = _shorthand_map(list(seen_rts.values()))   # {letter: endpoint}
        return next((ltr for ltr, ep in sh_map.items() if ep == endpoint), "—")

    def jid_tag(jid: str, endpoint: str, accel: str) -> str:
        sh = sh_for(endpoint)
        return f"[{sh:<2}/{accel:<4} {jid}]"

    def runtime_label(r: dict) -> str:
        ac = r.get("accelerator", "?")
        if ac in ("NONE", "VARIANT_UNSPECIFIED"):
            ac = "CPU"
        region = _parse_region_from_url(r.get("runtimeProxyInfo", {}).get("url", ""))
        desc = (_runtime_meta(r["endpoint"]) or {}).get("desc")
        desc_s = f"  — {_trunc(desc, 30)}" if desc else ""
        return f"{r['endpoint']}  ({ac}, {region}){desc_s}"

    def stream_one(jid: str, endpoint: str) -> None:
        path   = store.events_path(jid)
        sh     = sh_for(endpoint)
        col    = _color_for_letter(sh)
        prefix = f"[{sh:<2} {jid}]"
        # Wait for events file to appear
        while not path.exists() and not stop.is_set():
            time.sleep(0.1)
        if stop.is_set(): return

        line_buf = ""
        pos      = 0
        while not stop.is_set():
            rec = store.get_job(jid) or {}
            with path.open() as f:
                f.seek(pos); chunk = f.read(); pos = f.tell()
            for ev_line in chunk.splitlines():
                if not ev_line.strip(): continue
                try:
                    ev = json.loads(ev_line)
                except Exception:
                    continue
                text     = ev.get("text", "")
                is_err   = ev.get("type") in ("stderr", "error")
                # Treat \r as a line break too — tqdm and other progress libs
                # use \r-overwriting on stderr.  In multiplexed --follow the
                # \r would erase our [SH JID] prefix; promoting it to \n
                # gives each update its own attributable line.
                line_buf += text.replace("\r\n", "\n").replace("\r", "\n")
                while "\n" in line_buf:
                    ln, line_buf = line_buf.split("\n", 1)
                    if ln.strip():
                        # Prefix always in the runtime's letter color so you
                        # can tell which job an error came from.  Text turns
                        # red only for stderr/error events.
                        emit(prefix, ln, color=col,
                             text_color=91 if is_err else None)
            if rec.get("status") not in ("queued", "running"):
                if line_buf.strip():
                    emit(prefix, line_buf, color=col)
                return
            time.sleep(0.15)

    # ── seed from current state ──────────────────────────────────────────────
    try:
        initial_rts = list_runtimes(sess)
    except Exception as exc:
        emit("!", f"could not list runtimes: {exc}", color=91)
        initial_rts = []

    seen_rts = {r["endpoint"]: r for r in initial_rts}
    for r in initial_rts:
        if filter_endpoints is not None and r["endpoint"] not in filter_endpoints:
            continue
        sh = sh_for(r["endpoint"])
        emit("●", f"runtime active:    [{sh}]  {runtime_label(r)}",
             color=_color_for_letter(sh))

    for r in store.list_jobs():
        seen_jobs[r["job_id"]] = r["status"]
        if filter_endpoints is not None and r.get("endpoint") not in filter_endpoints:
            continue
        if r["status"] in ("queued", "running"):
            tag  = jid_tag(r["job_id"], r.get("endpoint", ""), r.get("accelerator", "?"))
            desc = r.get("desc")
            desc_s = f"  — {_trunc(desc, 30)}" if desc else ""
            emit("●", f"job in progress:   {tag}  on  {r.get('endpoint')}{desc_s}", color=36)
            started.add(r["job_id"])
            threading.Thread(
                target=stream_one,
                args=(r["job_id"], r.get("endpoint", "")),
                daemon=True,
            ).start()

    if filter_endpoints is not None:
        emit("●", f"follow filter:     {sorted(filter_endpoints)}", color=36)
    emit("●", "follow mode — Ctrl+C to exit", color=36)

    # ── main poll loop ───────────────────────────────────────────────────────
    REFRESH = 2.0
    last_token_check = time.time()

    def release_banner(ep: str, prev_record: dict) -> None:
        """Multi-line attention-grabbing banner for a released runtime."""
        meta   = _runtime_meta(ep) or {}
        reason = meta.get("released_by") or "unknown"
        accel  = (prev_record.get("accelerator") or meta.get("accelerator", "?"))
        if accel in ("NONE", "VARIANT_UNSPECIFIED"):
            accel = "CPU"
        region = _parse_region_from_url(prev_record.get("runtimeProxyInfo", {}).get("url", "")) or "?"
        assigned = meta.get("assigned_at")
        released = meta.get("released_at") or time.time()
        ran_h = ((released - assigned) / 3600) if assigned else None

        # Headline + reason text.
        if reason == "preempted":
            headline = "⚠  RUNTIME PREEMPTED BY COLAB  ⚠"
            human    = "Colab killed the runtime server-side (capacity-pressure preemption)"
            color    = 91   # red
        elif reason == "idle_timeout":
            t = meta.get("idle_timeout_min")
            headline = "✗  RUNTIME REAPED  ✗"
            human    = f"reached idle-timeout deadline (timeout={t}min)"
            color    = 33   # yellow
        elif reason == "user":
            headline = "✗  RUNTIME RELEASED BY YOU  ✗"
            human    = "explicit `colab --unassign`"
            color    = 33
        elif reason == "already_released":
            headline = "✗  RUNTIME ALREADY GONE  ✗"
            human    = "another client / Colab released it before we tried"
            color    = 33
        elif reason == "auto_reconcile":
            headline = "✗  RUNTIME LOST (reconciled)  ✗"
            human    = "endpoint disappeared from list_runtimes() — likely preempted"
            color    = 33
        else:
            headline = f"✗  RUNTIME RELEASED ({reason})  ✗"
            human    = ""
            color    = 33

        bar = "─" * 70
        ran_str = f"{ran_h:.2f}h" if ran_h is not None else "?"
        emit_block([
            "",
            bar,
            f"  {headline}",
            f"  endpoint:    {ep}",
            f"  accel:       {accel}        region:  {region}        ran:  {ran_str}",
            f"  reason:      {human}",
            bar,
        ], color=color)

    try:
        while not stop.is_set():
            time.sleep(REFRESH)

            # Refresh OAuth token periodically so long sessions don't 401
            if time.time() - last_token_check > 300:
                _ensure_fresh_token(sess)
                last_token_check = time.time()

            # Runtime delta
            try:
                cur = {r["endpoint"]: r for r in list_runtimes(sess)}
            except Exception:
                cur = seen_rts

            for ep in cur.keys() - seen_rts.keys():
                if filter_endpoints is not None and ep not in filter_endpoints: continue
                # `seen_rts` will gain ep this iteration — assign letter from updated map
                tmp_seen = {**seen_rts, ep: cur[ep]}
                _sh_map = _shorthand_map(list(tmp_seen.values()))
                _sh     = next((l for l, e in _sh_map.items() if e == ep), "?")
                emit("++", f"runtime assigned:  [{_sh}]  {runtime_label(cur[ep])}",
                     color=_color_for_letter(_sh))
            for ep in seen_rts.keys() - cur.keys():
                if filter_endpoints is not None and ep not in filter_endpoints: continue
                release_banner(ep, seen_rts[ep])
            seen_rts = cur

            # Job delta
            for r in store.list_jobs():
                jid = r["job_id"]
                old = seen_jobs.get(jid)
                new = r["status"]
                if old == new:
                    continue
                seen_jobs[jid] = new
                if filter_endpoints is not None and r.get("endpoint") not in filter_endpoints:
                    continue
                ep    = r.get("endpoint", "")
                accel = r.get("accelerator", "?")
                tag   = jid_tag(jid, ep, accel)
                desc  = r.get("desc")
                desc_s = f"  — {_trunc(desc, 30)}" if desc else ""
                if old is None:
                    emit("+", f"job queued:        {tag}  on  {ep}{desc_s}", color=32)
                elif new == "running":
                    emit("▶", f"job started:       {tag}{desc_s}", color=32)
                elif new == "cancelled":
                    emit("✗", f"job cancelled:     {tag}{desc_s}", color=33)
                elif new == "error":
                    emit("✖", f"job errored:       {tag}{desc_s}", color=91)
                elif new == "done":
                    emit("✔", f"job done:          {tag}{desc_s}", color=32)
                else:
                    emit("·", f"job {tag}: {old} → {new}{desc_s}", color=36)

                if new in ("queued", "running") and jid not in started:
                    started.add(jid)
                    threading.Thread(
                        target=stream_one,
                        args=(jid, ep),
                        daemon=True,
                    ).start()
    except KeyboardInterrupt:
        stop.set()
        emit("●", "follow stopped.", color=36)


def _spawn_watcher(job_id: str) -> int:
    """Detached subprocess that holds the WS open and writes events to disk."""
    import subprocess
    p = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--internal-watch", job_id],
        stdin  = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )
    return p.pid


# ── idle-timeout reaper ──────────────────────────────────────────────────────

def _reaper_lock_path(endpoint: str) -> Path:
    safe = endpoint.replace("/", "_")
    return _JOBS_DIR / f".reap_{safe}.lock"


def _spawn_reaper(endpoint: str) -> None:
    """Detach a `--internal-reap ENDPOINT` subprocess.

    The reaper acquires an exclusive flock on a per-endpoint lock file so we
    never have more than one running for the same runtime.  If another reaper
    is already counting, this is a cheap no-op.
    """
    import subprocess
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--internal-reap", endpoint],
        stdin  = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        start_new_session = True,
    )


def _runtime_has_active_jobs(endpoint: str) -> bool:
    store = _JobStore()
    for r in store.list_jobs():
        if r.get("endpoint") == endpoint and r.get("status") in ("queued", "running"):
            return True
    return False


def _reaper_alive(endpoint: str) -> bool:
    """True iff some other process currently holds the reaper lock."""
    import fcntl
    lock_path = _reaper_lock_path(endpoint)
    if not lock_path.exists():
        return False
    f = lock_path.open("a")
    try:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return False    # we got the lock → no live reaper
        except BlockingIOError:
            return True
    finally:
        f.close()


def _resurrect_reapers() -> None:
    """Spawn a reaper for each currently-alive runtime that has metadata but
    no live reaper.

    Called at the top of every CLI invocation so reapers killed by reboot /
    shutdown are respawned the next time the user runs ``colab``.  Skips:
      * Runtimes whose metadata says they've been released (kept around for
        historical `--runtimes --all` views, but they shouldn't get a reaper).
      * Runtimes that aren't in the live `list_runtimes()` API result —
        Colab killed them server-side (preemption etc.) and our local meta
        is stale; we mark them released and move on instead of spawning a
        zombie reaper that just 404s on every unassign attempt.
    """
    metas = _runtime_meta_all()
    if not metas:
        return
    candidates = [(ep, m) for ep, m in metas.items()
                  if not m.get("released_at") and (m.get("idle_timeout_min") or 0) > 0]
    if not candidates:
        return
    try:
        live_eps = {r["endpoint"] for r in list_runtimes(make_session())}
    except Exception:
        live_eps = None    # API hiccup — don't aggressively reconcile this pass
    now = time.time()
    for ep, m in candidates:
        if live_eps is not None and ep not in live_eps:
            # Endpoint not in the live API.  Could mean: (a) Colab killed the
            # runtime out from under us, OR (b) we just provisioned it and the
            # API hasn't propagated yet.  Give a 60s grace window after the
            # local `assigned_at` to avoid the false-positive case.
            if now - (m.get("assigned_at") or 0) < 60:
                continue
            try:
                _set_runtime_meta(ep, released_at=time.time(),
                                  released_by="auto_reconcile")
                _emit_event("runtime_released", endpoint=ep, reason="auto_reconcile")
            except Exception: pass
            continue
        if not _reaper_alive(ep):
            try:
                _spawn_reaper(ep)
            except Exception:
                pass


_REAPER_LOG = _JOBS_DIR / "reaper.log"


def _reap_log(endpoint: str, msg: str) -> None:
    """Append one timestamped decision line to reaper.log.  Best-effort."""
    try:
        _JOBS_DIR.mkdir(exist_ok=True)
        with _REAPER_LOG.open("a") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts}  pid={os.getpid()}  ep={endpoint[:38]:<38}  {msg}\n")
    except Exception:
        pass


def _internal_reap(endpoint: str) -> int:
    """Long-lived per-runtime daemon.

    Responsibilities:
      1. Keepalive ping ``/tun/m/{endpoint}/keep-alive/`` every ~30s so Colab's
         own idle disconnect (90 min on Pro) doesn't fire while we wait.
      2. Sleep until the runtime has been idle for its configured timeout,
         then unassign.

    Concurrency: at-most-one reaper per endpoint via ``fcntl.flock`` on a
    per-endpoint file.  Subsequent ``--internal-reap`` invocations exit
    immediately if another reaper holds the lock.

    Cancellation behavior:
      - ``colab --set-timeout`` mid-flight is picked up within 30s.
      - ``--idle-timeout 0`` shuts the reaper down.
      - When a new job starts on the runtime, the reaper exits; the next
        job-end will spawn a fresh one.
    """
    import fcntl
    lock_path = _reaper_lock_path(endpoint)
    lock_path.parent.mkdir(exist_ok=True)
    lock_f = lock_path.open("w")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _reap_log(endpoint, "spawn skipped — another reaper holds the lock")
        return 0

    _reap_log(endpoint, "spawn — acquired lock")
    sess           = make_session()
    keepalive_url  = f"{COLAB_DOMAIN}/tun/m/{endpoint}/keep-alive/"
    last_token_chk = time.time()

    ws_stop = threading.Event()
    last_kid: str | None = None
    try:
        for jr in sorted(_JobStore().list_jobs(),
                         key=lambda x: x.get("started", 0) or 0,
                         reverse=True):
            if jr.get("endpoint") == endpoint and jr.get("kernel_id"):
                last_kid = jr["kernel_id"]
                break
    except Exception:
        pass
    ws_thread = threading.Thread(
        target=_ws_keepalive_loop,
        args=(endpoint, last_kid, ws_stop),
        daemon=True,
        name=f"ws-keepalive-{endpoint[:12]}",
    )
    ws_thread.start()
    probe_thread = threading.Thread(
        target=_connections_probe_loop,
        args=(endpoint, ws_stop),
        daemon=True,
        name=f"ws-probe-{endpoint[:12]}",
    )
    probe_thread.start()
    resources_thread = threading.Thread(
        target=_resources_poll_loop,
        args=(endpoint, ws_stop),
        daemon=True,
        name=f"resources-poll-{endpoint[:12]}",
    )
    resources_thread.start()
    grpc_keepalive_thread = threading.Thread(
        target=_grpc_keepalive_loop,
        args=(endpoint, ws_stop),
        daemon=True,
        name=f"grpc-keepalive-{endpoint[:12]}",
    )
    grpc_keepalive_thread.start()

    def _ping() -> bool:
        """Hit the keep-alive endpoint.  Returns False iff the runtime is
        confirmed gone.  The keep-alive endpoint can return HTTP 400 during
        the runtime's warm-up window (before the proxy registers it) AS WELL
        AS for actually-preempted runtimes — so a 4xx alone is suspicious but
        not definitive.  We cross-check with `list_runtimes()` to confirm.
        """
        nonlocal last_token_chk
        suspicious = False
        try:
            r = sess.get(keepalive_url,
                         headers={"X-Colab-Tunnel": "Google"}, timeout=5)
            if r.status_code in (400, 404, 410):
                suspicious = True
        except Exception:
            pass
        if time.time() - last_token_chk > 600:
            try:
                _ensure_fresh_token(sess)
                last_token_chk = time.time()
            except Exception:
                pass
        if not suspicious:
            return True
        # Confirm via the assignments API — definitive signal.
        try:
            live = {r["endpoint"] for r in list_runtimes(sess)}
            return endpoint in live
        except Exception:
            return True    # transient API errors → don't trust the 4xx

    last_listcheck = 0.0
    try:
        while True:
            meta = _runtime_meta(endpoint) or {}
            timeout_min = meta.get("idle_timeout_min")
            if meta.get("released_at"):
                _reap_log(endpoint, "exit — meta has released_at (runtime was released by another path)")
                return 0
            if not timeout_min or timeout_min <= 0:
                # Timeout disabled — DON'T exit (would also kill the gRPC
                # keep-alive thread, which is what defeats Colab's ~20-min
                # A100 preemption).  Just skip the unassign-countdown logic
                # and keep the keep-alive threads running.  Re-check periodically
                # in case the user re-enables the timeout via --set-timeout.
                if ws_stop.wait(30):
                    return 0
                continue
            if _runtime_has_active_jobs(endpoint):
                # Don't exit — that would also kill the persistent ws-keepalive
                # thread, leaving Colab's idle scheduler with `connections == 0`
                # the moment the job's own _exec_sync WS closes (the gap that
                # causes ~20-min preemption on Pro+ A100s).  Sleep + re-check;
                # the idle countdown auto-pauses because base = max(ended_times)
                # is recomputed each iteration.
                if ws_stop.wait(30):
                    return 0
                continue
            # Periodic belt-and-suspenders check: every ~5 min during the sleep
            # loop, hit list_runtimes() and confirm our endpoint is still there.
            # This catches preemption even if keep-alive was misbehaving (e.g.
            # the Colab proxy's status-code semantics shift).
            if time.time() - last_listcheck > 300:
                if _is_preempted_double_check(sess, endpoint):
                    if (_runtime_meta(endpoint) or {}).get("released_at"):
                        _reap_log(endpoint, "exit — endpoint missing but released_at already set")
                        return 0
                    _reap_log(endpoint, "PREEMPTED — endpoint missing from list_runtimes() on double-check")
                    try:
                        _set_runtime_meta(endpoint, released_at=time.time(),
                                          released_by="preempted", preempted=True)
                        _emit_event("runtime_released", endpoint=endpoint, reason="preempted")
                    except Exception: pass
                    return 0
                last_listcheck = time.time()

            store       = _JobStore()
            ended_times = [r.get("ended", 0) for r in store.list_jobs()
                           if r.get("endpoint") == endpoint and r.get("ended")]
            base        = max(ended_times) if ended_times else meta.get("assigned_at", time.time())
            deadline    = base + timeout_min * 60

            wait = deadline - time.time()
            _reap_log(
                endpoint,
                f"countdown — timeout={timeout_min}min  base={time.strftime('%H:%M:%S', time.localtime(base))}"
                f"  deadline={time.strftime('%H:%M:%S', time.localtime(deadline))}  wait={wait:.0f}s",
            )
            if wait > 0:
                slept = 0
                preempted = False
                while slept < wait:
                    chunk = min(30, wait - slept)
                    time.sleep(chunk)
                    slept += chunk
                    if not _ping():
                        preempted = True
                        break
                    new_meta = _runtime_meta(endpoint) or {}
                    if (not new_meta.get("idle_timeout_min")) or _runtime_has_active_jobs(endpoint):
                        _reap_log(endpoint, "wake — meta changed or active job appeared, re-evaluating")
                        break
                if preempted:
                    if (_runtime_meta(endpoint) or {}).get("released_at"):
                        _reap_log(endpoint, "exit — keep-alive 4xx but released_at already set")
                        return 0
                    if not _is_preempted_double_check(sess, endpoint):
                        # `_ping` got a 4xx but list_runtimes() (twice) still
                        # shows the endpoint.  Transient blip, not real
                        # preemption — keep going.
                        _reap_log(endpoint, "wake — keep-alive 4xx but list_runtimes confirms alive; ignoring")
                        continue
                    _reap_log(endpoint, "PREEMPTED — keep-alive 4xx confirmed by double-check")
                    try:
                        _set_runtime_meta(endpoint,
                                          released_at=time.time(),
                                          released_by="preempted",
                                          preempted=True)
                        _emit_event("runtime_released", endpoint=endpoint, reason="preempted")
                    except Exception:
                        pass
                    return 0
                continue   # re-evaluate (timeout may have changed)

            if _runtime_has_active_jobs(endpoint):
                _reap_log(endpoint, "exit — active job appeared right before reap")
                return 0
            _reap_log(endpoint, "REAP — calling unassign_runtime (released_by=idle_timeout)")
            try:
                unassign_runtime(sess, endpoint, released_by="idle_timeout")
            except Exception as e:
                _reap_log(endpoint, f"REAP unassign raised {type(e).__name__}: {e}")
            return 0
    finally:
        ws_stop.set()
        ws_thread.join(timeout=8)
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


def _internal_watch(job_id: str) -> int:
    """Hidden mode used by detached --no-stream subprocess.

    Reads job meta, opens WS to its kernel, sends execute_request, persists
    every event to events.jsonl, and updates the index when done.
    """
    store = _JobStore()
    rec   = store.get_job(job_id)
    if rec is None:
        return 1

    sess  = make_session()
    fresh = _refresh_proxy_for_endpoint(sess, rec["endpoint"])
    if fresh is None:
        store.update_job(job_id, status="error", ended=time.time())
        store.append_event(job_id, {"type": "error",
            "text": f"[watcher] runtime {rec['endpoint']} no longer assigned\n"})
        return 2
    jurl, ptok = fresh

    job  = _Job(job_id, rec["code"], store=store)
    flag = threading.Event()
    auth = sess.headers.get("Authorization", "")
    store.update_job(job_id, status="running", started=time.time(),
                     jupyter_url=jurl, kernel_id=rec["kernel_id"])
    _write_job_pid(job_id)    # mark "this process is the handler"
    _emit_event("job_started", jid=job_id,
                endpoint=rec.get("endpoint"),
                accel=rec.get("accelerator"))

    # Keepalive thread for the duration of this watcher (default-on).  Detects
    # preemption mid-job and marks the runtime released.
    ka_stop = threading.Event()
    threading.Thread(
        target=_keepalive_loop,
        args=(sess, rec["endpoint"], ka_stop),
        daemon=True, name="dk-keepalive-watcher",
    ).start()

    # Resolve the env-injection choice the parent CLI saved on the record.
    # Default to True if absent (so old job records still get the new behavior).
    inject_choice = rec.get("inject_env", True)
    prelude       = _build_env_prelude(_resolve_env_for_job(inject_choice))
    try:
        _exec_sync(jurl, ptok, auth, rec["kernel_id"], rec["code"], job, flag,
                   timeout=24*3600, show=False, prelude=prelude)
        has_err = any(e.get("type") == "error" for e in job.events)
        job._finish("error" if has_err else "done")
        rc = 1 if has_err else 0
    except Exception as exc:
        job._append({"type": "error", "text": f"[watcher] {exc}\n"})
        job._finish("error")
        rc = 1
    finally:
        ka_stop.set()
        _clear_job_pid(job_id)
    # Idle-timeout reaper for the runtime this job ran on
    try:
        _spawn_reaper(rec["endpoint"])
    except Exception:
        pass
    return rc


def _internal_reattach(job_id: str, *, show: bool = False) -> int:
    """Reattach to a running kernel and capture NEW output for `job_id`
    until the kernel goes idle.

    Used by `colab --reattach JID` (foreground or via `--no-stream` daemon).
    Doesn't send an `execute_request` — just opens a Jupyter WS and listens
    to iopub.  Output emitted before this attached cannot be recovered
    (Jupyter doesn't buffer for disconnected clients), but everything from
    now on is appended to the same events.jsonl as the original watcher.

    Marks the job `done`/`error` and emits `job_done`/`job_error` when the
    kernel transitions busy → idle.  If the kernel is already idle on
    attach, marks the job done immediately.
    """
    store = _JobStore()
    rec = store.get_job(job_id)
    if rec is None:
        if show:
            print(f"[reattach] unknown job_id: {job_id!r}", file=sys.stderr)
        return 1
    if rec.get("status") in ("done", "error", "cancelled"):
        if show:
            print(f"[reattach] job {job_id} already terminal: {rec['status']}",
                  file=sys.stderr)
        return 0

    sess = make_session()
    fresh = _refresh_proxy_for_endpoint(sess, rec["endpoint"])
    if fresh is None:
        store.update_job(job_id, status="error", ended=time.time())
        store.append_event(job_id, {"type": "error",
            "text": f"[reattach] runtime {rec['endpoint']} no longer assigned\n"})
        _emit_event("job_error", jid=job_id, endpoint=rec.get("endpoint"),
                    accel=rec.get("accelerator"))
        return 2
    jurl, ptok = fresh
    _ensure_fresh_token(sess)
    auth = sess.headers.get("Authorization", "")

    # Probe: is the kernel idle already?
    try:
        r = sess.get(f"{jurl}/api/kernels",
                     headers={"X-Colab-Runtime-Proxy-Token": ptok, "X-Colab-Tunnel": "Google"},
                     timeout=10)
        kernels = r.json()
        tk = next((k for k in kernels if k.get("id") == rec.get("kernel_id")), None)
        if tk is None:
            store.update_job(job_id, status="error", ended=time.time())
            store.append_event(job_id, {"type": "error",
                "text": f"[reattach] kernel {rec.get('kernel_id')} not found on runtime\n"})
            _emit_event("job_error", jid=job_id, endpoint=rec.get("endpoint"),
                        accel=rec.get("accelerator"))
            return 3
        if tk.get("execution_state") == "idle":
            store.update_job(job_id, status="done", ended=time.time())
            store.append_event(job_id, {"type": "stderr",
                "text": "[reattach] kernel already idle on attach; job marked done "
                        "(any output between original disconnect and reattach is lost)\n"})
            _emit_event("job_done", jid=job_id, endpoint=rec.get("endpoint"),
                        accel=rec.get("accelerator"))
            if show: print(f"[reattach] kernel idle; job {job_id} marked done")
            return 0
    except Exception:
        pass

    session_id = str(uuid.uuid4())
    conn = _ws_handshake(jurl, ptok, auth, rec["kernel_id"], session_id, timeout=30)
    if show:
        print(f"[reattach] attached to kernel {rec['kernel_id'][:8]} on {rec['endpoint']}",
              flush=True)

    stop = threading.Event()
    def _hb_loop() -> None:
        while not stop.is_set():
            try:
                _ensure_fresh_token(sess)
                hb = {"header": {"msg_id": uuid.uuid4().hex,
                                 "msg_type": "kernel_info_request",
                                 "username": "reattach", "session": session_id,
                                 "date": "", "version": "5.3"},
                      "parent_header": {}, "metadata": {}, "content": {},
                      "channel": "shell", "buffers": []}
                conn.send(json.dumps(hb).encode())
            except Exception:
                pass
            if stop.wait(25):
                return
    threading.Thread(target=_hb_loop, daemon=True).start()

    _write_job_pid(job_id)
    seen_busy = False
    saw_error = False
    try:
        conn.settimeout(5)
        while True:
            try:
                op, payload = conn.recv_frame()
            except (TimeoutError, socket.timeout):
                continue
            except EOFError:
                store.append_event(job_id, {"type": "stderr",
                    "text": "[reattach] WebSocket EOF before kernel went idle; "
                            "output may be incomplete\n"})
                saw_error = True
                break
            except (OSError, ssl.SSLError) as exc:
                store.append_event(job_id, {"type": "stderr",
                    "text": f"[reattach] socket error {type(exc).__name__}: {exc}\n"})
                saw_error = True
                break
            if op == 0x8:
                break
            if op == 0x9:
                try: conn.send(payload, opcode=0xA)
                except Exception: pass
                continue
            try:
                msg = json.loads(payload)
            except Exception:
                continue
            mt = msg.get("msg_type", "")
            content = msg.get("content", {})
            if mt == "status":
                state = content.get("execution_state")
                if state == "busy":
                    seen_busy = True
                elif state == "idle" and seen_busy:
                    break
                continue
            ev: dict | None = None
            if mt == "stream":
                ev = {"type": content.get("name", "stdout"),
                      "text": content.get("text", "")}
            elif mt in ("display_data", "execute_result"):
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    ev = {"type": "result", "text": text + "\n",
                          "data": content.get("data", {})}
            elif mt == "error":
                saw_error = True
                tb = "\n".join(content.get("traceback", []))
                ev = {"type": "error", "text": tb + "\n",
                      "ename": content.get("ename", ""),
                      "evalue": content.get("evalue", "")}
            if ev:
                ev["_reattached"] = True
                store.append_event(job_id, ev)
                if show:
                    out = sys.stderr if ev["type"] in ("stderr", "error") else sys.stdout
                    out.write(ev.get("text", "")); out.flush()
    finally:
        stop.set()
        try: conn.close()
        except Exception: pass
        _clear_job_pid(job_id)

    final_status = "error" if saw_error else "done"
    store.update_job(job_id, status=final_status, ended=time.time())
    _emit_event(f"job_{final_status}", jid=job_id,
                endpoint=rec.get("endpoint"), accel=rec.get("accelerator"))
    if show:
        print(f"\n[reattach] job {job_id} → {final_status}")
    return 1 if saw_error else 0


# ── Colab Secrets probe ──────────────────────────────────────────────────────

# Names we always probe even if the user passes none.  Add yours here if it
# becomes a regular dependency.
_DEFAULT_SECRET_NAMES = [
    "HF_TOKEN", "HUGGINGFACE_TOKEN",
    "GITHUB_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "WANDB_API_KEY",
]


def _probe_secrets(extra_names: list[str]) -> int:
    """Connect to whichever runtime is handy and report each name's status.

    Output legend:
      OK     — secret accessible (from this notebook, with current toggles)
      GATED  — secret exists but notebook access not granted yet
      EMPTY  — no secret with that name in your account
      ERROR  — something else went wrong (e.g. network, kernel)
    """
    names = list(dict.fromkeys(list(extra_names) + _DEFAULT_SECRET_NAMES))

    sess = make_session()
    try:
        rts = list_runtimes(sess)
    except Exception as exc:
        print(f"[direct_kernel] could not list runtimes: {exc}", file=sys.stderr)
        return 1

    if not rts:
        nb = _NOTEBOOK_FILE.read_text().strip() if _NOTEBOOK_FILE.exists() else None
        print("[direct_kernel] No active runtime to probe with.")
        if nb:
            print( "Open the bridge notebook to view / toggle secrets:")
            print(f"  https://colab.research.google.com/drive/{nb}")
        else:
            print("Run `colab --test-cpu` once to create a notebook, then re-try.")
        return 1

    r    = rts[0]
    info = r.get("runtimeProxyInfo", {})
    jurl = info["url"].rstrip("/")
    ptok = info["token"]
    accel = _normalized_accel(r)
    print(f"[direct_kernel] probing on runtime {r['endpoint']} ({accel})", file=sys.stderr)

    kid = get_or_create_kernel(sess, jurl, ptok)

    probe_code = (
        "from google.colab import userdata\n"
        f"_NAMES = {names!r}\n"
        "_W = max(len(n) for n in _NAMES) + 2\n"
        "for n in _NAMES:\n"
        "    try:\n"
        "        v = userdata.get(n)\n"
        "        print(f'OK     {n:<{_W}} ({len(v)} chars)')\n"
        "    except userdata.NotebookAccessError:\n"
        "        print(f'GATED  {n:<{_W}} notebook access not granted')\n"
        "    except userdata.SecretNotFoundError:\n"
        "        print(f'EMPTY  {n:<{_W}} not defined in your account')\n"
        "    except Exception as e:\n"
        f"        print(f'ERROR  {{n:<{{_W}}}} {{type(e).__name__}}: {{e}}')\n"
    )

    job  = _Job(str(uuid.uuid4())[:8], probe_code)
    flag = threading.Event()
    auth = sess.headers.get("Authorization", "")
    _exec_sync(jurl, ptok, auth, kid, probe_code, job, flag, timeout=30, show=True)

    nb = _NOTEBOOK_FILE.read_text().strip() if _NOTEBOOK_FILE.exists() else None
    if nb:
        print()
        print("Manage secrets / toggle notebook access here:")
        print(f"  https://colab.research.google.com/drive/{nb}")
    return 0


# ── CLI ────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="colab",
        description="Run code on Google Colab runtimes via the VS Code extension API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
runtime management:
  colab --auth
  colab --runtimes                   # alias: --list  (table with letter shorthands)
  colab --unassign a                 # letter | accelerator | prefix | 'all'
  colab --unassign A100              # releases ALL A100s
  colab --set-timeout a 60           # change idle-timeout (minutes)
  colab --test-cpu

submit jobs (foreground; streams output here):
  colab -c "print(1+1)"
  colab --accelerator T4 -f experiments/train.py

submit jobs (detached; survives terminal exit):
  colab --no-stream -f experiments/long_train.py
  # → prints job_id; the job keeps running on Colab even after this exits

inspect / control submitted jobs:
  colab --jobs                # list all jobs
  colab --job-id JID          # replay & follow ONE job
  colab --latest              # replay & follow the newest job
  colab --follow              # multiplex EVERY running job (all runtimes)
  colab --follow A100         # multiplex only the A100 runtime
  colab --cancel JID          # graceful interrupt
  colab --live                # full-screen dashboard (press / for inline cmds)
  colab --log [N]             # tail the reaper daemon log (default last 200)
        """,
    )

    # runtime / auth
    parser.add_argument("--auth",          action="store_true",
                        help="Authenticate via Google OAuth (first-time setup)")
    parser.add_argument("--runtimes", "--list", action="store_true", dest="list",
                        help="Pretty table of active Colab runtimes (alias --list).  "
                             "With --all, also includes released ones from history.")
    parser.add_argument("--cost",     action="store_true",
                        help="Show credit balance, per-runtime cost so far, and "
                             "burn rate (units/hour).")
    parser.add_argument("--all",      action="store_true",
                        help="With --runtimes / --jobs, include historical / "
                             "released items rather than only currently-active ones.")
    parser.add_argument("--unassign", "-u", metavar="QUERY",
                        help="Release a runtime — accepts full endpoint, prefix, "
                             "substring, accelerator name (e.g. 'A100'), or 'all'")
    parser.add_argument("--assign",        action="store_true",
                        help="Just provision a runtime (per --accelerator) and "
                             "exit — no job submitted.  Reaper / keepalive / "
                             "idle-timeout still apply.")
    parser.add_argument("--test-cpu",      action="store_true",
                        help="Quick connectivity test on a CPU runtime")
    parser.add_argument("--accelerator", "-a", default="CPU", metavar="GPU",
                        help="CPU, T4, L4, A100, H100, G4 (default: CPU). "
                             "Used only when --runtime is NOT given; with "
                             "--runtime, the existing runtime's accelerator "
                             "is used as-is.")
    parser.add_argument("--runtime",     "-r", metavar="RUNTIME",
                        help="Submit to a specific *already-assigned* runtime, "
                             "by letter (a/b/…), accelerator name, endpoint "
                             "prefix, or unique substring.  Recommended over "
                             "--accelerator when you want to control which "
                             "runtime gets used.")
    parser.add_argument("--high-ram",      action="store_true",
                        help="Request the high-RAM machine shape")
    parser.add_argument("--idle-timeout",  type=int, metavar="MIN", default=None,
                        help="Auto-unassign this runtime after MIN minutes of "
                             "idle (no jobs queued/running). Default 30, 0 disables. "
                             "Settable any time later via --set-timeout.")
    parser.add_argument("--set-timeout",   nargs=2, metavar=("RUNTIME", "MIN"),
                        help="Change the idle-timeout (minutes) of an existing "
                             "runtime. RUNTIME accepts the same shorthand as "
                             "--unassign (a / A100 / 'all' / endpoint prefix).")
    parser.add_argument("--notebook-url",  action="store_true",
                        help="Print the URL of the bridge's Drive notebook "
                             "(open it in a browser to manage Colab secrets).")
    parser.add_argument("--repair-notebook", action="store_true",
                        help="Overwrite the saved notebook on Drive with a "
                             "minimal valid .ipynb body. Use if the URL shows "
                             "'corrupted or not a valid notebook file'.")
    parser.add_argument("--secrets",       nargs="*", metavar="NAME",
                        default=None,
                        help="Probe a runtime for accessible Colab secrets. "
                             "With no NAMEs, checks a default list; pass "
                             "names to check specific ones (e.g. --secrets HF_TOKEN).")

    # local-secrets management (.env)
    parser.add_argument("--env",           action="store_true",
                        help="List the names defined in colab_bridge/.env "
                             "(values not shown).")
    parser.add_argument("--env-set",       metavar="NAME[=VALUE]",
                        help="Set / overwrite a secret in colab_bridge/.env. "
                             "With NAME alone, prompt for the value (hidden "
                             "input).  With NAME=VALUE, set directly (note: "
                             "value lands in your shell history).  Combine "
                             "with --from-file to load multi-line values "
                             "from disk.")
    parser.add_argument("--from-file",     metavar="PATH",
                        help="With --env-set NAME, read the value verbatim "
                             "from PATH (multi-line OK).")
    parser.add_argument("--env-rm",        metavar="NAME",
                        help="Remove NAME from colab_bridge/.env.")
    parser.add_argument("--env-show",      metavar="NAME",
                        help="Print the value of NAME (use with care; for "
                             "scripting / sanity checks).")

    # per-job env-injection control
    parser.add_argument("--no-inject-env", action="store_true",
                        help="Skip injecting .env into this job (default is "
                             "to inject every key).")
    parser.add_argument("--inject-env",    nargs="+", metavar="NAME",
                        help="Inject only these specific names from .env "
                             "(default is all).")
    parser.add_argument("--keepalive",     action="store_true",
                        help=argparse.SUPPRESS)    # legacy; on by default now
    parser.add_argument("--no-keepalive",  action="store_true",
                        help="Disable the per-job keep-alive thread (default: on). "
                             "The thread pings /tun/m/{endpoint}/keep-alive/ every "
                             "60s during a foreground job and stops at job-end.")
    parser.add_argument("--timeout",       type=float, default=3600,
                        help="Foreground execution timeout in seconds (default: 3600)")
    parser.add_argument("--file-id",       metavar="DRIVE_FILE_ID",
                        help="Use a specific Drive file ID for the notebook")

    # job inspection / control
    parser.add_argument("--jobs",          nargs="?", const="ALL", default=None,
                        metavar="RUNTIME",
                        help="List jobs.  Pass an optional RUNTIME shorthand "
                             "(letter / accelerator / 'all' / endpoint substring) "
                             "to restrict to one runtime.")
    parser.add_argument("--json",          action="store_true",
                        help="With --jobs / --runtimes, emit raw JSON instead of a table")
    parser.add_argument("--job-id",        metavar="JID",
                        help="Replay history + follow the named job to completion")
    parser.add_argument("--latest",        nargs="?", const="ALL", default=None,
                        metavar="RUNTIME",
                        help="Replay + follow the most recently RAN job (skips "
                             "queued).  Optional RUNTIME restricts the search.")
    parser.add_argument("--follow",        nargs="?", const="ALL", default=None,
                        metavar="RUNTIME",
                        help="Multiplexed live tail of all running jobs + "
                             "runtime/job lifecycle events. Pass an optional "
                             "shorthand (accelerator, endpoint prefix, or "
                             "substring) to restrict to one runtime")
    parser.add_argument("--cancel",        metavar="JID",
                        help="Send a graceful interrupt to the running job")

    # Alerts: event log subscriber commands
    parser.add_argument("--watch",         action="store_true",
                        help="Tail the event log (events.jsonl).  Use --type / "
                             "--jid / --endpoint to filter, --once for first match. "
                             f"Event types: {_EVENT_TYPES_HELP}.")
    parser.add_argument("--type",          action="append", metavar="TYPE",
                        help="With --watch, only show this event type "
                             "(repeatable).")
    parser.add_argument("--once",          action="store_true",
                        help="With --watch / --wait-for-*, exit after first match.")
    parser.add_argument("--from-start",    action="store_true",
                        help="With --watch, replay history from the beginning of "
                             "the log instead of tailing only new events.")
    parser.add_argument("--wait-for-job",  metavar="JID",
                        help="Block until the named job leaves running/queued. "
                             "Exit 0 on done, 1 on error/cancelled, 2 if unknown.")
    parser.add_argument("--wait-for-runtime", metavar="QUERY",
                        help="Block until the named runtime is released "
                             "(reason in stdout). Same shorthand as --unassign.")
    parser.add_argument("--events",        action="store_true",
                        help="Live-stream every event in human-readable form "
                             "(prettier than --watch).  Ctrl+C to exit.")
    parser.add_argument("--event",         nargs="+", metavar=("TYPE", "K=V"),
                        help="Emit a contrived event into events.jsonl.  "
                             "First arg is the event type (e.g. `wake`, "
                             "`agent_done`, `user_signal`); subsequent args are "
                             "optional `key=value` fields attached to the "
                             "event.  Useful for waking up agents that watch "
                             "events.jsonl: e.g. `colab --watch --type wake --once` "
                             "in one terminal, `colab --event wake reason=manual` "
                             "in another.")
    parser.add_argument("--live",          action="store_true",
                        help="Full-screen live dashboard of active runtimes + "
                             "open jobs + burn rate.  Updates every ~1s. "
                             "Ctrl+C to exit.  Press / inside the dashboard "
                             "to run inline commands (see /help).")
    parser.add_argument("--log",           nargs="?", const=200, default=None,
                        type=int, metavar="N",
                        help="Print the tail of the reaper daemon log "
                             "(reaper.log) — useful when chasing preemption / "
                             "keepalive issues.  Default N=200 lines; pair "
                             "with --all to dump the entire file.")

    # submission
    parser.add_argument("--no-stream",     action="store_true",
                        help="Submit and detach: print job_id, exit, job runs in background")

    parser.add_argument("--reattach", metavar="JID",
                        help="Reattach to a job whose original watcher died "
                             "(captures NEW stdout/stderr/result/error events "
                             "into the existing events.jsonl until the kernel "
                             "goes idle).  Output emitted before reattach is "
                             "unrecoverable.  With --no-stream, runs detached.")
    parser.add_argument("--status", metavar="JID",
                        help="Print full metadata + connection status for one job "
                             "(connected / stale / disconnected). Same data as "
                             "`/status JID` in --live.")

    # internal (hidden)
    parser.add_argument("--internal-watch",    metavar="JID", help=argparse.SUPPRESS)
    parser.add_argument("--internal-reap",     metavar="ENDPOINT", help=argparse.SUPPRESS)
    parser.add_argument("--internal-reattach", metavar="JID", help=argparse.SUPPRESS)

    parser.add_argument("--desc",  "-d", metavar="TEXT", default=None,
                        help="Optional one-line description.  With --assign, "
                             "tags the new runtime.  With -c / -f / --no-stream, "
                             "tags the job.  Shows up in --jobs, --runtimes, "
                             "--cost, --follow, --live, etc.  Recommended but "
                             "not required.")

    src = parser.add_mutually_exclusive_group()
    src.add_argument("-c", "--code", metavar="CODE", help="Python code string to execute")
    src.add_argument("-f", "--file", metavar="FILE", help="Python file to execute")

    args = parser.parse_args()

    # ── internal watcher (hidden) ──────────────────────────────────────────────
    if args.internal_watch:
        sys.exit(_internal_watch(args.internal_watch))

    # ── internal reaper (hidden) ───────────────────────────────────────────────
    if args.internal_reap:
        sys.exit(_internal_reap(args.internal_reap))

    # ── internal reattach daemon (hidden) ──────────────────────────────────────
    if args.internal_reattach:
        sys.exit(_internal_reattach(args.internal_reattach, show=False))

    # ── --status JID ───────────────────────────────────────────────────────────
    if args.status:
        store_ = _JobStore()
        rec = store_.get_job(args.status)
        if rec is None:
            print(f"unknown job_id: {args.status!r}", file=sys.stderr)
            sys.exit(1)
        fmt_t = lambda t: (time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(t)) if t else "—")
        conn = _job_connection_status(rec)
        conn_color = {"connected":     "\033[32m",
                      "stale":         "\033[33m",
                      "disconnected":  "\033[31m",
                      "queued":        "\033[36m"}.get(conn, "\033[90m")
        status_disp = "queued" if conn == "queued" else rec.get("status", "?")
        print(f"job {rec['job_id']}: \033[1m{status_disp}\033[0m")
        print(f"  endpoint:    {rec.get('endpoint','?')}")
        print(f"  accelerator: {rec.get('accelerator','?')}")
        print(f"  started:     {fmt_t(rec.get('started'))}")
        print(f"  ended:       {fmt_t(rec.get('ended'))}")
        print(f"  desc:        {rec.get('desc') or '—'}")
        line = f"  connection:  {conn_color}●\033[0m {conn}"
        if conn == "disconnected":
            line += f"  \033[2m(use `colab --reattach {rec['job_id']}`)\033[0m"
        print(line)
        code = (rec.get("code") or "").strip()
        code1 = (code.replace("\n", "↵")[:120] if code else "—")
        print(f"  code:        {code1}")
        sys.exit(0)

    # ── --reattach JID (foreground or detached) ────────────────────────────────
    if args.reattach:
        if args.no_stream:
            import subprocess
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 "--internal-reattach", args.reattach],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
            )
            print(f"[direct_kernel] reattach daemon spawned for {args.reattach}")
            sys.exit(0)
        sys.exit(_internal_reattach(args.reattach, show=True))

    # Best-effort: respawn any reapers that were killed by a reboot/shutdown.
    # Cheap (a few Popen calls); no-ops if a reaper is already alive.
    try:
        _resurrect_reapers()
    except Exception:
        pass

    # ── auth ───────────────────────────────────────────────────────────────────
    if args.auth:
        do_auth()
        print("[direct_kernel] Auth complete.  Try --test-cpu next.", flush=True)
        return

    # ── notebook URL (for managing Colab Secrets in the browser) ──────────────
    if args.notebook_url:
        if not _NOTEBOOK_FILE.exists():
            print("[direct_kernel] No saved notebook yet.  Run any submit "
                  "(e.g. --test-cpu) once to create one.", file=sys.stderr)
            sys.exit(1)
        nb = _NOTEBOOK_FILE.read_text().strip()
        print(f"https://colab.research.google.com/drive/{nb}")
        return

    # ── repair a corrupted notebook ───────────────────────────────────────────
    if args.repair_notebook:
        if not _NOTEBOOK_FILE.exists():
            print("[direct_kernel] No saved notebook to repair.", file=sys.stderr)
            sys.exit(1)
        nb = _NOTEBOOK_FILE.read_text().strip()
        sess = make_session()
        repair_notebook(sess, nb)
        print(f"[direct_kernel] Repaired notebook {nb}.")
        print(f"  https://colab.research.google.com/drive/{nb}")
        return

    # ── probe Colab Secrets ────────────────────────────────────────────────────
    if args.secrets is not None:
        sys.exit(_probe_secrets(args.secrets or []))

    # ── local secrets management (.env) ────────────────────────────────────────
    if args.env:
        env = _load_env()
        if not env:
            print("[direct_kernel] no secrets defined.  Try `colab --env-set NAME`.",
                  file=sys.stderr)
            return
        for k in sorted(env):
            v = env[k]
            print(f"  {k}  ({len(v)} chars)" if v else f"  {k}  (empty)")
        return

    if args.env_set:
        env = _load_env()
        spec = args.env_set
        if args.from_file:
            name = spec.strip()
            value = Path(args.from_file).read_text()
        elif "=" in spec:
            name, _, value = spec.partition("=")
            name = name.strip()
        else:
            name = spec.strip()
            import getpass
            value = getpass.getpass(f"value for {name}: ")
        if not name:
            print("[direct_kernel] empty name", file=sys.stderr); sys.exit(1)
        env[name] = value
        _save_env(env)
        print(f"[direct_kernel] {name} set ({len(value)} chars).")
        return

    if args.env_rm:
        env = _load_env()
        if args.env_rm not in env:
            print(f"[direct_kernel] {args.env_rm} not in .env.", file=sys.stderr)
            sys.exit(1)
        del env[args.env_rm]
        _save_env(env)
        print(f"[direct_kernel] {args.env_rm} removed.")
        return

    if args.env_show:
        env = _load_env()
        if args.env_show not in env:
            print(f"[direct_kernel] {args.env_show} not in .env.", file=sys.stderr)
            sys.exit(1)
        print(env[args.env_show])
        return

    # ── list runtimes ──────────────────────────────────────────────────────────
    if args.list:
        sess = make_session()
        rts  = list_runtimes(sess)
        _heal_stale_released(rts); _stamp_missing_letters(rts)    # backfill letters for legacy entries
        if args.all:
            # Augment with released entries from local meta
            active_eps = {r["endpoint"] for r in rts}
            for ep, meta in _runtime_meta_all().items():
                if ep in active_eps:
                    continue
                if not meta.get("released_at"):
                    continue
                rts.append({
                    "endpoint": ep,
                    "accelerator": meta.get("accelerator", "?"),
                    "machineShape": "SHAPE_DEFAULT",
                    "_released_at": meta.get("released_at"),
                    "_assigned_at": meta.get("assigned_at"),
                    "runtimeProxyInfo": {},
                })
        if args.json:
            print(json.dumps(rts, indent=2))
        else:
            _print_runtimes_table(rts)
        return

    # ── cost / billing ────────────────────────────────────────────────────────
    if args.cost:
        sess     = make_session()
        balance  = _account_balance(sess)
        if balance is not None:
            _record_balance(balance)
        rts      = list_runtimes(sess)
        _heal_stale_released(rts); _stamp_missing_letters(rts)
        observed = _observed_rate_per_hour()
        _print_cost_table(rts, balance, observed)
        return

    # ── unassign runtime(s) ────────────────────────────────────────────────────
    if args.unassign:
        sess = make_session()
        try:
            eps = _resolve_endpoints(args.unassign, list_runtimes(sess))
        except ValueError as exc:
            print(f"[direct_kernel] {exc}", file=sys.stderr)
            sys.exit(1)
        for ep in eps:
            print(json.dumps(unassign_runtime(sess, ep), indent=2))
        return

    # ── change idle-timeout on an existing runtime ─────────────────────────────
    if args.set_timeout:
        runtime_q, minutes_s = args.set_timeout
        try:
            minutes = int(minutes_s)
        except ValueError:
            print(f"[direct_kernel] --set-timeout MIN must be integer, got {minutes_s!r}",
                  file=sys.stderr); sys.exit(1)
        sess = make_session()
        try:
            eps = _resolve_endpoints(runtime_q, list_runtimes(sess))
        except ValueError as exc:
            print(f"[direct_kernel] {exc}", file=sys.stderr); sys.exit(1)
        for ep in eps:
            _set_runtime_meta(ep, idle_timeout_min=minutes)
            label = "off" if minutes == 0 else f"{minutes} min"
            print(f"[direct_kernel] {ep}  →  idle timeout: {label}")
            # Refresh reaper so it picks up the new value (or shuts itself down)
            try:
                _spawn_reaper(ep)
            except Exception:
                pass
        return

    # ── job listing ────────────────────────────────────────────────────────────
    if args.jobs:
        _reconcile_orphan_jobs()
        try:
            _rts_tmp = list_runtimes(make_session()); _heal_stale_released(_rts_tmp); _stamp_missing_letters(_rts_tmp); del _rts_tmp
        except Exception:
            pass
        store = _JobStore()
        jobs  = store.list_jobs()
        if args.jobs != "ALL":
            # Explicit runtime filter wins.
            jobs = _filter_jobs_by_runtime(jobs, args.jobs)
        elif args.all:
            # --jobs --all → entire history (the old default).
            pass
        else:
            # New default — only jobs whose endpoint is currently assigned.
            try:
                active_eps = {r["endpoint"] for r in list_runtimes(make_session())}
                jobs = [j for j in jobs if j.get("endpoint") in active_eps]
            except Exception:
                pass    # API blip — fall back to showing everything
        if args.json:
            print(json.dumps(jobs, indent=2))
        else:
            _print_jobs_table(jobs)
            if not args.all and args.jobs == "ALL":
                print("\n(showing only jobs on currently-active runtimes — "
                      "use `colab --jobs --all` for full history)")
        return

    # ── attach to a specific job ───────────────────────────────────────────────
    if args.job_id or args.latest:
        _reconcile_orphan_jobs()
        store = _JobStore()
        if args.latest:
            jobs = [j for j in store.list_jobs() if j.get("status") != "queued"]
            if args.latest != "ALL":
                jobs = _filter_jobs_by_runtime(jobs, args.latest)
            if not jobs:
                print("No matching ran jobs recorded.")
                return
            rec = max(jobs, key=lambda r: r.get("started", 0))
            jid = rec["job_id"]
        else:
            jid = args.job_id
            rec = store.get_job(jid)
            if not rec:
                print(f"[direct_kernel] unknown job_id: {jid!r}", file=sys.stderr)
                sys.exit(1)
        print(f"[direct_kernel] attaching to {jid} on runtime {rec['endpoint']} "
              f"({rec.get('accelerator', '?')})", file=sys.stderr, flush=True)
        for _ in _stream_from_disk(store, jid, show=True):
            pass
        rec = store.get_job(jid) or rec
        print(f"\n[direct_kernel] {jid} finished — status={rec.get('status')}",
              file=sys.stderr, flush=True)
        return

    # ── alerts: event log subscriber ───────────────────────────────────────────
    if args.watch:
        types = set(args.type) if args.type else None
        sys.exit(_watch_events(types=types,
                               jid=args.job_id,
                               once=args.once,
                               from_start=args.from_start))

    if args.wait_for_job:
        sys.exit(_wait_for_job(args.wait_for_job))

    if args.wait_for_runtime:
        sys.exit(_wait_for_runtime(args.wait_for_runtime))

    if args.events:
        sys.exit(_stream_events_human())

    if args.event:
        # Emit one contrived event line: type + optional key=value fields.
        ev_type = args.event[0]
        fields: dict = {}
        for kv in args.event[1:]:
            k, _, v = kv.partition("=")
            if not k:
                print(f"[direct_kernel] ignoring malformed --event arg: {kv!r}",
                      file=sys.stderr)
                continue
            fields[k] = v
        _emit_event(ev_type, **fields)
        print(f"[direct_kernel] emitted event type={ev_type!r}"
              + (f"  fields={fields}" if fields else ""))
        sys.exit(0)

    if args.live:
        sys.exit(_live_dashboard())

    # ── reaper-log tail ────────────────────────────────────────────────────────
    if args.log is not None:
        if not _REAPER_LOG.exists():
            print(f"[direct_kernel] no reaper log yet at {_REAPER_LOG}")
            return
        lines = _REAPER_LOG.read_text().splitlines()
        out = lines if args.all else lines[-args.log:]
        print("\n".join(out))
        return

    # ── follow: live multiplexer over all runtimes + jobs ─────────────────────
    if args.follow:
        if args.follow == "ALL":
            _follow_all()
        else:
            try:
                eps = _resolve_endpoints(args.follow, list_runtimes(make_session()))
            except ValueError as exc:
                print(f"[direct_kernel] {exc}", file=sys.stderr)
                sys.exit(1)
            _follow_all(filter_endpoints=set(eps))
        return

    # ── cancel a running job ───────────────────────────────────────────────────
    if args.cancel:
        store = _JobStore()
        rec   = store.get_job(args.cancel)
        if not rec:
            print(f"[direct_kernel] unknown job_id: {args.cancel!r}", file=sys.stderr)
            sys.exit(1)
        if rec["status"] not in ("queued", "running"):
            print(f"[direct_kernel] job already {rec['status']}, nothing to cancel.")
            return
        # Queued (in store) OR running-but-not-actually-executing (the watcher
        # has flipped status="running" but the kernel is still busy with a
        # prior cell): in either case the kernel hasn't picked our request
        # up, so an interrupt would target whatever's currently running —
        # not our job.  Just remove from queue.
        if rec["status"] == "queued" or _job_connection_status(rec) == "queued":
            store.update_job(args.cancel, status="cancelled", ended=time.time())
            store.append_event(args.cancel, {"type": "stderr",
                "text": "\n[direct_kernel] cancelled before start (was queued)\n"})
            print(f"[direct_kernel] queued job {args.cancel} removed from queue "
                  f"(no kernel interrupt needed)")
            return
        sess  = make_session()
        fresh = _refresh_proxy_for_endpoint(sess, rec["endpoint"])
        if fresh is None:
            print(f"[direct_kernel] runtime {rec['endpoint']} no longer assigned; "
                  f"marking cancelled.", file=sys.stderr)
            store.update_job(args.cancel, status="cancelled", ended=time.time())
            return
        jurl, ptok = fresh
        # Mark cancelled BEFORE sending the interrupt.  The kernel-side
        # error-frame round-trip via WebSocket is sometimes faster than our
        # local HTTP interrupt round-trip; if we updated after, the watcher's
        # `_finish("error")` could see status=running and overwrite to
        # "error" before `_finish`'s preserve-cancelled check would help.
        store.update_job(args.cancel, status="cancelled", ended=time.time())
        store.append_event(args.cancel, {"type": "stderr",
            "text": "\n[direct_kernel] cancelled by --cancel\n"})
        ok = _do_interrupt(sess, jurl, ptok, rec["kernel_id"])
        print(f"[direct_kernel] interrupt sent to runtime {rec['endpoint']} "
              f"(kernel {rec['kernel_id'][:8]}…) — {'ok' if ok else 'http error'}")
        return

    # ── just assign a runtime, no job ─────────────────────────────────────────
    if args.assign:
        accel = args.accelerator
        print(f"[direct_kernel] Assigning (accelerator={accel})…", flush=True)
        sess           = make_session()
        fid            = args.file_id or get_or_create_notebook(sess)
        # allow_pool=True so repeated `--assign` calls give us NEW runtimes
        # via fresh throwaway notebooks, instead of returning the same one.
        jurl, ptok, ep = assign_runtime(sess, fid, accel, args.high_ram,
                                        idle_timeout_min=args.idle_timeout,
                                        allow_pool=True,
                                        desc=args.desc)
        # Eagerly create a kernel so the runtime is fully warm and ready.
        kid = get_or_create_kernel(sess, jurl, ptok)
        # Identifying summary so the user can spot which runtime is theirs
        # at a glance — short letter shorthand, accelerator, region, kernel
        # ID (the closest thing to a "GPU id" — uniquely names this GPU
        # session within the user's account).
        meta = _runtime_meta(ep) or {}
        sh = meta.get("letter") or "?"
        region = _parse_region_from_url(jurl) or "—"
        print(f"[direct_kernel] yours → \033[1msh={sh}  accel={accel.upper()}  "
              f"region={region}  kernel={kid}\033[0m", flush=True)
        print(ep)
        return

    # ── determine code to run ──────────────────────────────────────────────────
    accel = "CPU" if args.test_cpu else args.accelerator

    if args.test_cpu:
        code = (
            "import sys, platform\n"
            "print('Python:', sys.version)\n"
            "print('Platform:', platform.platform())\n"
            "try:\n"
            "    import torch\n"
            "    print('PyTorch:', torch.__version__)\n"
            "    print('CUDA:', torch.cuda.is_available())\n"
            "    if torch.cuda.is_available():\n"
            "        print('GPU:', torch.cuda.get_device_name(0))\n"
            "except ImportError:\n"
            "    print('torch not installed')\n"
            "print('[direct_kernel] OK')\n"
        )
    elif args.code:
        code = args.code
    elif args.file:
        code = Path(args.file).read_text()
    else:
        parser.print_help()
        sys.exit(1)

    # ── connect ────────────────────────────────────────────────────────────────
    sess = make_session()
    if args.runtime:
        # Target a specific already-assigned runtime — bypass `assign_runtime`
        # (which is keyed on our notebook id and would just return whatever
        # runtime that notebook has).
        try:
            rts   = list_runtimes(sess)
            picks = _resolve_endpoints(args.runtime, rts)
        except ValueError as exc:
            print(f"[direct_kernel] {exc}", file=sys.stderr); sys.exit(1)
        if len(picks) > 1:
            print(f"[direct_kernel] {args.runtime!r} matches multiple runtimes; "
                  "be more specific (use a letter):\n  " + "\n  ".join(picks),
                  file=sys.stderr); sys.exit(1)
        ep        = picks[0]
        rt_rec    = next(r for r in rts if r["endpoint"] == ep)
        info      = rt_rec["runtimeProxyInfo"]
        jurl      = info["url"].rstrip("/")
        ptok      = info["token"]
        accel     = _normalized_accel(rt_rec)
        # Refresh idle-timeout if the user explicitly asked for one.
        if args.idle_timeout is not None:
            _set_runtime_meta(ep, idle_timeout_min=args.idle_timeout)
        print(f"[direct_kernel] Targeting existing runtime {ep} ({accel})", flush=True)
    else:
        print(f"[direct_kernel] Connecting (accelerator={accel})…", flush=True)
        fid            = args.file_id or get_or_create_notebook(sess)
        # If --desc is given alongside a job submit (no --runtime), it
        # describes the JOB; we pass desc=None to assign_runtime so we don't
        # accidentally label the runtime with the job's description.
        jurl, ptok, ep = assign_runtime(sess, fid, accel, args.high_ram,
                                        idle_timeout_min=args.idle_timeout)
    kid = get_or_create_kernel(sess, jurl, ptok)

    # Resolve env-injection choice: explicit --inject-env list > --no-inject-env > all
    if args.no_inject_env:
        inject_choice: bool | list[str] = False
    elif args.inject_env:
        inject_choice = list(args.inject_env)
    else:
        inject_choice = True

    # Register job in the store before kicking off execution
    store = _JobStore()
    jid   = str(uuid.uuid4())[:8]
    store.add_job({
        "job_id":      jid,
        "status":      "queued",
        "endpoint":    ep,
        "accelerator": accel.upper(),
        "jupyter_url": jurl,
        "kernel_id":   kid,
        "code":        code,
        "desc":        args.desc,
        "inject_env":  inject_choice,
        "started":     time.time(),
        "ended":       None,
    })
    print(f"[direct_kernel] job {jid} queued on runtime {ep} ({accel.upper()})",
          flush=True)

    # ── detached submission ───────────────────────────────────────────────────
    if args.no_stream:
        pid = _spawn_watcher(jid)
        print(f"[direct_kernel] yours → \033[1mjid={jid}  pid={pid}  runtime={ep}\033[0m  "
              f"\033[2m(kill {pid} to stop the local watcher; the job runs on Colab)\033[0m",
              file=sys.stderr, flush=True)
        print(jid)        # <-- job_id on stdout, as old client.py did
        return
    # Foreground: the local "handler" is this CLI process itself.  Print its
    # PID too so the user can match it to ps/Activity Monitor and Ctrl+C the
    # right window when running multiple foreground jobs from different shells.
    print(f"[direct_kernel] yours → \033[1mjid={jid}  pid={os.getpid()}  runtime={ep}\033[0m",
          file=sys.stderr, flush=True)

    # ── foreground streaming ───────────────────────────────────────────────────
    # Keepalive is on by default — pings the runtime every 60s so Pro/Free
    # idle timers don't fire mid-job.  Pass --no-keepalive to disable.
    ka_stop: threading.Event | None = None
    if not args.no_keepalive:
        ka_stop = threading.Event()
        threading.Thread(
            target=_keepalive_loop,
            args=(sess, ep, ka_stop),
            daemon=True, name="dk-keepalive",
        ).start()

    job            = _Job(jid, code, store=store)
    interrupt_flag = threading.Event()
    auth_header    = sess.headers.get("Authorization", "")

    store.update_job(jid, status="running", started=time.time())
    _write_job_pid(jid)
    _emit_event("job_started", jid=jid, endpoint=ep, accel=accel.upper())

    prelude = _build_env_prelude(_resolve_env_for_job(inject_choice))

    t0 = time.time()
    try:
        _exec_sync(
            jurl, ptok, auth_header, kid, code,
            job, interrupt_flag,
            timeout=args.timeout, show=True, prelude=prelude,
        )
        has_err = any(e.get("type") == "error" for e in job.events)
        job._finish("error" if has_err else "done")
    finally:
        _clear_job_pid(jid)
    elapsed = time.time() - t0

    if ka_stop:
        ka_stop.set()

    # Idle-timeout reaper for this runtime
    try:
        _spawn_reaper(ep)
    except Exception:
        pass

    print(
        f"\n[direct_kernel] {jid} finished in {elapsed:.1f}s — {'ERROR' if has_err else 'OK'}",
        file=sys.stderr, flush=True,
    )
    sys.exit(1 if has_err else 0)


if __name__ == "__main__":
    _cli()
