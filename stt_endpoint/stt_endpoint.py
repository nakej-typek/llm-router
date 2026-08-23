#!/usr/bin/env python3
"""
OWUI Speech-to-Text backend — the SAME Gemini transcription JP already uses in
voice_windows_reference.py (Claudoslav Voice), exposed as an OpenAI-Whisper-compatible
HTTP endpoint so Open WebUI's mic button can use it instead of its built-in Whisper.

Why: OWUI supports configuring STT to any OpenAI-compatible endpoint (Admin Settings ->
Audio -> Speech-to-Text -> Engine: OpenAI, custom Base URL). This server implements just
enough of that contract:

  POST /v1/audio/transcriptions   multipart/form-data, field "file" = audio blob
    -> 200 {"text": "<transcript>"}

MODEL/PROMPT/fallback logic mirrors voice_windows_reference.py exactly (JP wants "úplně
ten samý" — same model, same prompt, same behavior), just swapping the transport (HTTP
multipart in, JSON out) instead of a global hotkey + local mic capture.

Audio from a browser mic is usually webm/opus — transcoded to WAV via ffmpeg first (safe,
known-good format for Gemini's inlineData) rather than trusting the raw browser MIME type.

Article 1: same free GOOGLE_API_KEY as the rest of the stack. stdlib + ffmpeg only.
"""
import base64
import email
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "gemini-3.1-flash-lite"            # same as voice_windows_reference.py
FALLBACK_MODEL = "gemini-3-flash-preview"  # same fallback, for when MODEL is overloaded
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
HOST = os.environ.get("OWUI_STT_HOST", "127.0.0.1")
PORT = int(os.environ.get("OWUI_STT_PORT", "8902"))
HERMES_ENV = os.path.expanduser("~/.hermes/.env")

# Identical prompt to voice_windows_reference.py (JP tuned this; keep it exact).
PROMPT = """Přepiš tuto nahrávku doslovně.

Mluvčí je Čech a míchá češtinu s anglickými technickými výrazy — anglická
slova nech anglicky, nepřekládej je a nepřepisuj foneticky.
Časté názvy: Claude, Claude Code, Claudoslav, Gemini, Python, commit, prompt.

Vrať POUZE přepsaný text — žádný úvod, komentář, uvozovky ani markdown.
Když v nahrávce není žádná řeč, vrať prázdný řetězec."""


def read_key():
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    try:
        with open(HERMES_ENV, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def to_wav(raw_bytes):
    """ffmpeg: whatever the browser sent (webm/opus, ogg, ...) -> 16kHz mono WAV,
    a format we know Gemini's inlineData accepts cleanly."""
    with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as fin:
        fin.write(raw_bytes)
        in_path = fin.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", out_path],
            capture_output=True, timeout=30, check=True)
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def _call(model, wav_b64, key):
    body = {
        "contents": [{"parts": [
            {"text": PROMPT},
            {"inlineData": {"mimeType": "audio/wav", "data": wav_b64}},
        ]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
    }
    url = f"{GEMINI_BASE}/models/{model}:generateContent?key={key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.load(r)
    return (resp["candidates"][0]["content"]["parts"][0]["text"] or "").strip()


def transcribe(wav_bytes):
    key = read_key()
    if not key:
        raise RuntimeError("chybí GOOGLE_API_KEY")
    wav_b64 = base64.b64encode(wav_bytes).decode()
    last = None
    for model in (MODEL, MODEL, FALLBACK_MODEL):  # same retry pattern as voice_windows_reference.py
        try:
            return _call(model, wav_b64, key)
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in (429, 500, 503):
                raise
            time.sleep(1.5)
    raise last


def _read_body(rfile, headers):
    """stdlib http.server only understands Content-Length — it does NOT decode
    Transfer-Encoding: chunked. OWUI's backend (httpx) streams the multipart upload
    chunked, so we have to decode that ourselves or every request 400s with the
    leftover chunk-size bytes misread as the next request line."""
    if (headers.get("Transfer-Encoding") or "").lower() == "chunked":
        chunks = []
        while True:
            size_line = rfile.readline().strip()
            size = int(size_line.split(b";")[0], 16)  # ignore chunk extensions
            if size == 0:
                rfile.readline()  # trailing CRLF after the terminating 0-chunk
                break
            chunks.append(rfile.read(size))
            rfile.readline()  # CRLF after each chunk's data
        return b"".join(chunks)
    length = int(headers.get("Content-Length", 0))
    return rfile.read(length)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {"ok": True, "model": MODEL})
        elif self.path.rstrip("/") == "/v1/models":
            self._json(200, {"data": [{"id": MODEL}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/transcriptions":
            self._json(404, {"error": "not found; use POST /v1/audio/transcriptions"})
            return
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            self._json(400, {"error": "expected multipart/form-data with a 'file' field"})
            return
        try:
            body = _read_body(self.rfile, self.headers)
            # `cgi` was removed in Python 3.13+ — parse multipart via the still-supported
            # `email` package instead: reconstruct a MIME message and pull out the part
            # whose Content-Disposition name is "file".
            header_bytes = f"Content-Type: {ctype}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            msg = email.message_from_bytes(header_bytes + body)
            audio_bytes = None
            for part in msg.get_payload() or []:
                name = part.get_param("name", header="Content-Disposition")
                if name == "file":
                    audio_bytes = part.get_payload(decode=True)
                    break
        except Exception as e:
            self._json(400, {"error": f"bad multipart body: {e}"})
            return
        if not audio_bytes:
            self._json(400, {"error": "missing 'file' field"})
            return
        t0 = time.time()
        try:
            wav = to_wav(audio_bytes)
            text = transcribe(wav)
        except Exception as e:
            print(f"[stt] ERROR after {time.time()-t0:.1f}s: {e}", flush=True)
            self._json(502, {"error": str(e)})
            return
        print(f"[stt] ok in {time.time()-t0:.1f}s: {text[:60]!r}", flush=True)
        self._json(200, {"text": text})

    def log_message(self, fmt, *args):
        # temporarily verbose (debugging a 400) — shows raw HTTP parsing errors that
        # happen before do_POST ever runs
        print("[http]", fmt % args, flush=True)


def main():
    print(f"OWUI STT (Gemini {MODEL}) na http://{HOST}:{PORT}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
