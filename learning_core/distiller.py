#!/usr/bin/env python3
"""
Learning core — distiller.

Reads new raw conversation text (append-only, raw/*.md), folds durable facts into
wiki/profile.md following schema/schema.md, using free Gemini flash-lite. The wiki is
the single source of truth; this script never touches Hermes's internal memory.

Constitution: FREE only (flash-lite, has the most generous free quota). stdlib only,
no deps, no paid calls. Readable markdown corpus (Articles B & D). See CONSTITUTION.md.

Run once:  python3 distiller.py          (processes new raw, updates the profile)
Idempotent: tracks per-file byte offsets in .distiller_state.json; only new bytes are
processed. Backs up the profile before each rewrite.
"""

import glob
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

import profile_history   # lokální modul, ne stdlib

BASE = os.environ.get("LEARNING_CORE_DIR", os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE, "raw")
WIKI = os.path.join(BASE, "wiki", "profile.md")
SCHEMA = os.path.join(BASE, "schema", "schema.md")
STATE = os.path.join(BASE, ".distiller_state.json")

HERMES_ENV = os.path.expanduser("~/.hermes/.env")
MODEL = "gemini-flash-lite-latest"          # free, sustainable (per-model quota)
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

MIN_NEW_CHARS = 200        # don't bother distilling tiny additions; batch them up
MAX_INPUT_CHARS = 900_000  # cap prompt size; if exceeded, use the most recent slice.
# Bumped 2026-07-30 (was 100_000): CC session ingest (JP wants ALL chats read, not just
# OWUI) can add 600K+ chars in one run, and offsets advance past whatever's silently
# truncated here — anything cut is lost to distillation FOREVER, not deferred. Gemini
# flash-class models have large context windows (hundreds of thousands of tokens), so
# 900K chars (~200-250K tokens) fits comfortably with headroom.
KEEP_BACKUPS = 5


def log(msg):
    print(f"[distiller {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_api_key():
    key = os.environ.get("GOOGLE_API_KEY")
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


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE)


def gather_new_raw(state):
    """Return (new_text, new_offsets) for bytes appended since last run."""
    new_offsets = dict(state)
    chunks = []
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.md"))):
        name = os.path.basename(path)
        if name.lower() == "readme.md":
            continue
        size = os.path.getsize(path)
        seen = state.get(name, 0)
        if size <= seen:
            new_offsets[name] = size
            continue
        with open(path, "rb") as f:
            f.seek(seen)
            data = f.read()
        chunks.append(f"### from {name}\n" + data.decode("utf-8", "replace"))
        new_offsets[name] = size
    text = "\n\n".join(chunks).strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[-MAX_INPUT_CHARS:]
    return text, new_offsets


def gemini(key, prompt):
    url = f"{GEMINI_BASE}/models/{MODEL}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}}
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=90, context=ssl.create_default_context()) as r:
        data = json.loads(r.read().decode())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip() + "\n"


def backup_profile():
    """Snapshot profile.md before each rewrite.

    Backups live in wiki/.backups/, NOT next to profile.md. Reason (2026-08-10): Hermes
    reads the wiki with search_files/read_file, and five sibling `profile.md.bak.*` files
    meant its searches kept matching stale snapshots — it pulled a superseded backup as a
    result while answering JP. A dot-directory keeps the snapshots (they are the only
    defence against a bad distillation) without polluting what the agent sees.
    """
    if not os.path.exists(WIKI):
        return
    bak_dir = os.path.join(os.path.dirname(WIKI), ".backups")
    os.makedirs(bak_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = os.path.join(bak_dir, f"{os.path.basename(WIKI)}.bak.{stamp}")
    with open(WIKI, encoding="utf-8") as src, open(bak, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    backups = sorted(glob.glob(os.path.join(bak_dir, f"{os.path.basename(WIKI)}.bak.*")))
    for old in backups[:-KEEP_BACKUPS]:
        try:
            os.remove(old)
        except OSError:
            pass


def write_profile(text):
    # PROVENANCE (2026-08-12). backup_profile() keeps KEEP_BACKUPS snapshots against a
    # 30-minute timer, so the record of WHEN a claim entered the profile is only hours
    # deep. That mattered on 2026-08-10: "lives near Hrnčířská in Ponava" (a district)
    # became "visits the Ponávka indoor pool" (a swimming pool JP had never heard of),
    # and it was traceable only because a 07-26 baseline happened to survive.
    #
    # Why the obvious audit does not work: once a fabrication has been echoed back by JP
    # it IS in the corpus, so grepping raw/ for support clears exactly the claims that
    # already looped. The only question that separates invention from fact is "did the
    # corpus say this BEFORE the profile did?" — which needs history that rotation deletes.
    #
    # Hooked here and not in backup_profile() because the diff needs old AND new; at
    # backup time only the old exists. append() never raises (see its docstring) —
    # losing one log line costs provenance for one run, losing the distillation costs
    # the run.
    old = ""
    if os.path.exists(WIKI):
        with open(WIKI, encoding="utf-8") as f:
            old = f.read()
    profile_history.append(WIKI, old, text)
    tmp = WIKI + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, WIKI)


PROMPT_TEMPLATE = """You maintain a living profile of a person, JP, distilled from his
conversations with his assistant. Follow the SCHEMA below exactly.

===== SCHEMA =====
{schema}

===== CURRENT PROFILE =====
{profile}

===== NEW CONVERSATION EXCERPTS (since last run) =====
{new_text}

===== TASK =====
Integrate any durable new facts from the excerpts into the profile, following every
rule in the SCHEMA. Return the COMPLETE updated profile.md as markdown only — no
preamble, no explanation, no code fences."""


def main():
    key = read_api_key()
    if not key:
        log("no GOOGLE_API_KEY — aborting.")
        return 1

    state = load_state()
    new_text, new_offsets = gather_new_raw(state)
    if len(new_text) < MIN_NEW_CHARS:
        log(f"only {len(new_text)} new chars (< {MIN_NEW_CHARS}); nothing to distill.")
        return 0

    try:
        schema = open(SCHEMA, encoding="utf-8").read()
        profile = open(WIKI, encoding="utf-8").read()
    except OSError as e:
        log(f"cannot read schema/profile: {e}")
        return 1

    prompt = PROMPT_TEMPLATE.format(schema=schema, profile=profile, new_text=new_text)
    log(f"distilling {len(new_text)} new chars via {MODEL}…")
    try:
        out = gemini(key, prompt)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            log("Gemini 429 (quota) — skipping this run, offsets unchanged. Retry next timer.")
            return 0
        log(f"Gemini HTTP {e.code}: {e.read().decode()[:200]}")
        return 1
    except Exception as e:
        log(f"Gemini error: {e}")
        return 1

    updated = strip_fences(out)
    if len(updated) < 50 or "##" not in updated:
        log("suspicious model output (too short / no headings) — NOT overwriting profile.")
        return 1

    backup_profile()
    write_profile(updated)
    save_state(new_offsets)   # only advance offsets after a successful write
    log(f"profile updated ({len(updated)} chars). offsets saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
