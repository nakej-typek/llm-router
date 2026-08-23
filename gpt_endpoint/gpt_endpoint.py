#!/usr/bin/env python3
"""
GPT interface endpoint for the AI router (W-022).

Exposes JP's Codex CLI (ChatGPT SUBSCRIPTION sign-in, NOT a per-token API key ->
Article 1 fine) as a tiny HTTP endpoint on the TAILNET, so the router (currently on
JP's Windows box) can reach it. GPT analog of the status panel + `claude -p` pattern.

Contract (W-022):
  POST /ask   body {"prompt": "...", "model": "<optional>"}
              -> 200 {"answer": "...", "model": "<optional>", "elapsed_sec": <float>}
              -> 4xx/5xx {"error": "..."}
  GET  /health -> {"ok": true, "logged_in": <bool>, "codex_version": "...", "bind": "..."}

Mechanism: shells out to `codex exec --sandbox read-only --skip-git-repo-check
--color never -o <tmp>` and returns the final agent message (-o writes JUST the last
message, so we don't parse streamed logs). Runs in a scratch cwd outside Syncthing.

Security: binds the TAILSCALE IP only (never 0.0.0.0), tailnet-only, no public exposure.
Optional shared token: if GPT_ENDPOINT_TOKEN is set, require `Authorization: Bearer <t>`.
stdlib only.

Env:
  GPT_ENDPOINT_HOST   bind host (default 127.0.0.1; systemd drop-in sets the tailnet IP)
  GPT_ENDPOINT_PORT   bind port (default 8901)
  GPT_ENDPOINT_TOKEN  optional bearer token; if unset, tailnet trust only
  CODEX_BIN           codex binary (default ~/.hermes/node/bin/codex, fallback PATH `codex`)
  CODEX_TIMEOUT       per-call seconds (default 180)
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("GPT_ENDPOINT_HOST", "127.0.0.1")
PORT = int(os.environ.get("GPT_ENDPOINT_PORT", "8901"))
TOKEN = os.environ.get("GPT_ENDPOINT_TOKEN", "").strip()
TIMEOUT = int(os.environ.get("CODEX_TIMEOUT", "180"))
MAX_PROMPT = 32000

WORKDIR = os.path.expanduser("~/.local/share/ai_router/gpt_workdir")

# Usage ledger for the guard (W-023): ONE WRITER PER FILE to avoid Syncthing
# conflict copies — the Arch endpoint owns usage.gpt.jsonl; the Windows router
# owns usage.router.jsonl. The guard reads the union.
QUOTA_DIR = os.environ.get("AI_ROUTER_QUOTA_DIR") or os.environ.get(
    "QUOTA_DIR", os.path.expanduser("~/syncthing/archlinux/ai_router/quota"))
GPT_LEDGER = os.path.join(QUOTA_DIR, "usage.gpt.jsonl")
GPT_SLUG = os.environ.get("GPT_SLUG", "openai/gpt-free")


def _parse_resets_at(text):
    """Best-effort: pull a reset time from a Codex rate-limit message. Returns an
    epoch float or None. Handles 'try again in 12m/3600s' relative hints."""
    t = (text or "").lower()
    m = re.search(r"try again in\s+(\d+)\s*(second|sec|s|minute|min|m|hour|h)", t)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        mult = 1 if unit.startswith("s") else 3600 if unit.startswith("h") else 60
        return round(time.time() + n * mult, 3)
    return None


def append_ledger(event, **extra):
    """Append one JSONL line to usage.gpt.jsonl (single-writer -> Syncthing-safe)."""
    try:
        os.makedirs(QUOTA_DIR, exist_ok=True)
        rec = {"ts": round(time.time(), 3), "slug": GPT_SLUG, "event": event}
        rec.update({k: v for k, v in extra.items() if v is not None})
        with open(GPT_LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[ledger] append failed: {e}")


def codex_bin():
    cand = os.environ.get("CODEX_BIN") or os.path.expanduser("~/.hermes/node/bin/codex")
    if os.path.exists(cand):
        return cand
    return shutil.which("codex") or cand


def codex_env():
    env = dict(os.environ)
    # make sure the codex bin dir is on PATH for any node shims it spawns
    binp = os.path.dirname(codex_bin())
    env["PATH"] = binp + os.pathsep + env.get("PATH", "")
    return env


def is_logged_in():
    try:
        r = subprocess.run([codex_bin(), "login", "status"],
                           capture_output=True, text=True, timeout=15, env=codex_env())
        return "not logged in" not in (r.stdout + r.stderr).lower()
    except (OSError, subprocess.SubprocessError):
        return False


def codex_version():
    try:
        r = subprocess.run([codex_bin(), "--version"],
                           capture_output=True, text=True, timeout=15, env=codex_env())
        return (r.stdout or r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# substrings that mean "Codex 5h usage window is spent" (not a generic failure) ->
# the router should treat this like a rate-limit cooldown and rotate to free Gemini.
_RATE_LIMIT_HINTS = ("usage limit", "rate limit", "rate_limit", "quota",
                     "too many requests", "429", "try again later", "resets")


def _looks_rate_limited(text):
    t = (text or "").lower()
    return any(h in t for h in _RATE_LIMIT_HINTS)


def run_codex(prompt, model=None):
    """Run one non-interactive Codex session.
    Returns (answer, err, rate_limited: bool)."""
    os.makedirs(WORKDIR, exist_ok=True)
    fd, last_msg = tempfile.mkstemp(prefix="codex_last_", suffix=".txt", dir=WORKDIR)
    os.close(fd)
    cmd = [codex_bin(), "exec", "--skip-git-repo-check", "--sandbox", "read-only",
           "--color", "never", "-o", last_msg]
    if model:
        cmd += ["-m", str(model)]
    cmd += [prompt]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT,
                              cwd=WORKDIR, env=codex_env())
    except subprocess.TimeoutExpired:
        return None, f"codex timed out after {TIMEOUT}s", False
    except OSError as e:
        return None, f"failed to launch codex: {e}", False
    try:
        with open(last_msg, encoding="utf-8") as f:
            answer = f.read().strip()
    except OSError:
        answer = ""
    finally:
        try:
            os.remove(last_msg)
        except OSError:
            pass
    if not answer:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return None, f"codex returned no message (exit {proc.returncode}): {tail}", \
            _looks_rate_limited(tail)
    return answer, None, False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self):
        if not TOKEN:
            return True
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {TOKEN}"

    def log_message(self, *a):
        pass  # quiet; systemd journal captures our own prints

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, "logged_in": is_logged_in(),
                             "codex_version": codex_version(),
                             "bind": f"{HOST}:{PORT}"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/ask":
            self._send(404, {"error": "not found; use POST /ask"})
            return
        if not self._authed():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, OSError):
            self._send(400, {"error": "invalid JSON body"})
            return
        prompt = (data.get("prompt") or "").strip()
        if not prompt:
            self._send(400, {"error": "missing 'prompt'"})
            return
        if len(prompt) > MAX_PROMPT:
            self._send(400, {"error": f"prompt too long (>{MAX_PROMPT} chars)"})
            return
        t0 = time.time()
        answer, err, rate_limited = run_codex(prompt, data.get("model"))
        elapsed = round(time.time() - t0, 2)
        if err:
            # 429 = 5h window spent -> router should cooldown + rotate to free Gemini.
            # 502 = generic Codex failure.
            code = 429 if rate_limited else 502
            print(f"[ask] {'RATE-LIMITED' if rate_limited else 'ERROR'} in {elapsed}s: {err}")
            if rate_limited:
                # authoritative live signal to the guard: window is spent
                append_ledger("rate_limited", resets_at=_parse_resets_at(err),
                              note=err[-160:])
            self._send(code, {"error": err, "rate_limited": rate_limited,
                              "elapsed_sec": elapsed})
            return
        append_ledger("call", elapsed_sec=elapsed)  # count this use for the guard
        print(f"[ask] ok in {elapsed}s ({len(answer)} chars)")
        out = {"answer": answer, "elapsed_sec": elapsed}
        if data.get("model"):
            out["model"] = data["model"]
        self._send(200, out)


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"gpt_endpoint on http://{HOST}:{PORT}  codex={codex_bin()}  "
          f"logged_in={is_logged_in()}  token={'on' if TOKEN else 'off'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
