#!/usr/bin/env python3
"""
One-time BULK historical import of JP's exported AI chat history → learning wiki.

W-017. FREE Gemini (model != voice's gemini-3.1-flash-lite so it can NEVER starve JP's
push-to-talk dictation — separate per-model quota bucket). Map-reduce, resumable, 429-safe,
paced over hours/days by a systemd timer. Gemini does the distillation, NOT Claude.

Public/private split per fact:
  public  → technical, professional, projects, skills, tools, interests, logistics → wiki/profile.md
  private → relationships, wellbeing, mental/emotional health, professional support → wiki/profile_private.md
  (when unsure → private; private = INSIGHT not raw negativity; neutral, non-therapy.)

Layout:
  SOURCES (read-only): ~/syncthing/archlinux/ai_router_import/memory_imports/{claude,chatgpt,gemini}
  WORK (LOCAL only, sensitive — NOT synced): ~/.local/share/ai_router/bulk_import/
      records.jsonl        normalized conversations (extract phase)
      facts_public.jsonl   per-batch extracted public facts (map phase)
      facts_private.jsonl  per-batch extracted private facts (map phase)
      state.json           progress: phase + next batch index
  OUTPUT: learning_core/wiki/profile.md (merge) + learning_core/wiki/profile_private.md (merge)

Invocation: `bulk_import.py step`  — does a bounded slice of work then exits (timer-paced).
            `bulk_import.py status` — prints progress.
stdlib only.
"""

import glob
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

HOME = os.path.expanduser("~")
SRC = os.path.join(HOME, "syncthing/archlinux/ai_router_import/memory_imports")
LC = os.path.join(HOME, "syncthing/archlinux/ai_router/learning_core")
WIKI_PUBLIC = os.path.join(LC, "wiki", "profile.md")
WIKI_PRIVATE = os.path.join(LC, "wiki", "profile_private.md")
SCHEMA = os.path.join(LC, "schema", "schema.md")

WORK = os.path.join(HOME, ".local/share/ai_router/bulk_import")   # LOCAL, sensitive
RECORDS = os.path.join(WORK, "records.jsonl")
FACTS_PUB = os.path.join(WORK, "facts_public.jsonl")
FACTS_PRIV = os.path.join(WORK, "facts_private.jsonl")
STATE = os.path.join(WORK, "state.json")

HERMES_ENV = os.path.join(HOME, ".hermes/.env")
MODEL = "gemini-3.5-flash-lite"          # != gemini-3.1-flash-lite (voice)
# ⚠️ 2026-08-10 (A-051/A-053): "separate quota bucket" was asserted here and it is NOT
# established. Free-tier quotas are per MODEL ROW and per PROJECT, and the alias
# gemini-flash-lite-latest (distiller + status-panel canary) resolves onto one of the
# versioned lite rows — which one is UNVERIFIED. Under either mapping the alias collides
# with one of these two models. Proven mechanism: a 429 for gemini-flash-latest named
# "model: gemini-3.6-flash", so aliases bill against versioned rows. Do not rely on the
# separation this line used to claim without checking the AI Studio dashboard.
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

PER_CONV_CAP = 6000        # chars kept per conversation (head+tail) — keep batches sane
BATCH_CHARS = 40000        # target input chars per Gemini map call (packs many convos)
BATCHES_PER_RUN = 4        # bounded work per invocation (every 30 min) → ~192 calls/day,
                           # under the ~250/day free cap with headroom. Paces over ~1-2 days.
SLEEP_BETWEEN = 4.0        # throttle between calls (voice is a DIFFERENT model bucket; be gentle)

PRIVATE_GUARDRAILS = (
    "PRIVATE = relationships, wellbeing, mental/emotional health, personal struggles, "
    "professional support (therapy/coaching), anything sensitive. When unsure, mark PRIVATE. "
    "For PRIVATE facts: distill to INSIGHT (durable patterns, triggers, what helps, context) — "
    "do NOT copy raw negative-thought dumps; stay neutral and non-judgmental; no therapy language."
)


def log(m):
    print(f"[bulk_import {datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def read_key():
    k = os.environ.get("GOOGLE_API_KEY")
    if k:
        return k.strip()
    try:
        for line in open(HERMES_ENV, encoding="utf-8"):
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
        return {"phase": "extract", "map_batch": 0}


def save_state(s):
    os.makedirs(WORK, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)
    os.replace(tmp, STATE)


def cap(text):
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= PER_CONV_CAP:
        return text
    half = PER_CONV_CAP // 2
    return text[:half] + "\n…[trimmed]…\n" + text[-half:]


# ---------- EXTRACT ----------
def extract_claude(records):
    path = os.path.join(SRC, "claude", "conversations.json")
    if not os.path.exists(path):
        return
    data = json.load(open(path, encoding="utf-8"))
    for c in data:
        msgs = c.get("chat_messages") or []
        lines = []
        for m in msgs:
            sender = "JP" if m.get("sender") == "human" else "AI"
            txt = m.get("text") or ""
            if not txt and isinstance(m.get("content"), list):
                txt = " ".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
            if txt.strip():
                lines.append(f"{sender}: {txt.strip()}")
        if lines:
            records.append({"src": "claude", "title": c.get("name") or "",
                            "date": str(c.get("created_at") or ""), "text": cap("\n".join(lines))})


def extract_chatgpt(records):
    for path in sorted(glob.glob(os.path.join(SRC, "chatgpt", "conversations-*.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            log(f"chatgpt parse fail {os.path.basename(path)}: {e}")
            continue
        for c in data:
            mapping = c.get("mapping") or {}
            nodes = []
            for node in mapping.values():
                msg = node.get("message")
                if not msg:
                    continue
                role = (msg.get("author") or {}).get("role")
                if role not in ("user", "assistant"):
                    continue
                content = msg.get("content") or {}
                parts = content.get("parts") or []
                txt = " ".join(p for p in parts if isinstance(p, str))
                if txt.strip():
                    nodes.append((msg.get("create_time") or 0, "JP" if role == "user" else "AI", txt.strip()))
            nodes.sort(key=lambda x: x[0])
            lines = [f"{who}: {txt}" for _, who, txt in nodes]
            if lines:
                records.append({"src": "chatgpt", "title": c.get("title") or "",
                                "date": str(c.get("create_time") or ""), "text": cap("\n".join(lines))})


def extract_gemini(records):
    for path in sorted(glob.glob(os.path.join(SRC, "gemini", "*.html"))):
        try:
            raw = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
        text = re.sub(r"(?s)<[^>]+>", "\n", raw)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text).strip()
        # chunk the activity log into PER_CONV_CAP-sized records
        for i in range(0, len(text), PER_CONV_CAP):
            chunk = text[i:i + PER_CONV_CAP].strip()
            if len(chunk) > 200:
                records.append({"src": "gemini", "title": os.path.basename(path),
                                "date": "", "text": chunk})


def do_extract():
    os.makedirs(WORK, exist_ok=True)
    records = []
    extract_claude(records)
    extract_chatgpt(records)
    extract_gemini(records)
    with open(RECORDS, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"extracted {len(records)} conversation records → records.jsonl")
    return len(records)


# ---------- MAP ----------
def load_records():
    return [json.loads(l) for l in open(RECORDS, encoding="utf-8")] if os.path.exists(RECORDS) else []


def make_batches(records):
    batches, cur, size = [], [], 0
    for r in records:
        rt = len(r["text"])
        if cur and size + rt > BATCH_CHARS:
            batches.append(cur); cur, size = [], 0
        cur.append(r); size += rt
    if cur:
        batches.append(cur)
    return batches


def gemini_call(key, prompt, max_tokens=4096):
    url = f"{GEMINI_BASE}/models/{MODEL}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as r:
        data = json.loads(r.read().decode())
    return data["candidates"][0]["content"]["parts"][0]["text"]


MAP_PROMPT = """You are building a durable profile of a person, JP, from his past AI chat logs.
From the conversations below, extract STABLE, DURABLE facts about JP (identity, work, projects,
skills, tools, interests, logistics, preferences, and personal/emotional context). Skip transient
chatter, one-off task specifics, and anything not durably about JP. Never include secrets
(API keys, passwords, precise home address).

Classify EACH fact as public or private:
- PUBLIC: technical, professional, projects, skills, tools, interests, logistics, preferences.
- {guardrails}

Output STRICT JSON only, no prose, no code fences:
{{"public": ["short english fact", ...], "private": ["short english fact", ...]}}
Empty arrays if nothing durable.

===== CONVERSATIONS =====
{convos}"""


def parse_json_facts(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return [], []
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return [], []
    pub = [str(x).strip() for x in d.get("public", []) if str(x).strip()]
    priv = [str(x).strip() for x in d.get("private", []) if str(x).strip()]
    return pub, priv


def do_map(key, state):
    records = load_records()
    batches = make_batches(records)
    total = len(batches)
    i = state.get("map_batch", 0)
    if i >= total:
        state["phase"] = "reduce"; save_state(state)
        log(f"map complete ({total} batches). → reduce")
        return
    done_this_run = 0
    while i < total and done_this_run < BATCHES_PER_RUN:
        batch = batches[i]
        convos = "\n\n---\n\n".join(
            f"[{r['src']}] {r['title']}\n{r['text']}" for r in batch)
        prompt = MAP_PROMPT.format(guardrails=PRIVATE_GUARDRAILS, convos=convos)
        try:
            out = gemini_call(key, prompt)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log(f"429 on batch {i}/{total} — pausing, resume next run.")
                save_state(state)
                return
            log(f"HTTP {e.code} on batch {i} — skipping batch.")
            i += 1; state["map_batch"] = i; save_state(state); continue
        except Exception as e:
            log(f"error on batch {i} ({e}) — pausing.")
            save_state(state)
            return
        pub, priv = parse_json_facts(out)
        with open(FACTS_PUB, "a", encoding="utf-8") as f:
            for fact in pub:
                f.write(json.dumps({"b": i, "fact": fact}, ensure_ascii=False) + "\n")
        with open(FACTS_PRIV, "a", encoding="utf-8") as f:
            for fact in priv:
                f.write(json.dumps({"b": i, "fact": fact}, ensure_ascii=False) + "\n")
        log(f"batch {i}/{total}: +{len(pub)} public, +{len(priv)} private")
        i += 1; state["map_batch"] = i; save_state(state)
        done_this_run += 1
        if i < total and done_this_run < BATCHES_PER_RUN:
            time.sleep(SLEEP_BETWEEN)
    if i >= total:
        state["phase"] = "reduce"
    save_state(state)


# ---------- REDUCE ----------
def _load_facts(path):
    if not os.path.exists(path):
        return []
    seen, out = set(), []
    for l in open(path, encoding="utf-8"):
        try:
            fact = json.loads(l)["fact"].strip()
        except Exception:
            continue
        k = fact.lower()
        if fact and k not in seen:
            seen.add(k); out.append(fact)
    return out


REDUCE_PUBLIC_PROMPT = """You maintain JP's PUBLIC profile. Merge the NEW facts below into the
current profile, following the SCHEMA. Integrate — do not lose existing content; refine and
deduplicate. Return the COMPLETE updated profile.md as markdown only, no code fences.

===== SCHEMA =====
{schema}

===== CURRENT PROFILE =====
{profile}

===== NEW FACTS (bulk import) =====
{facts}"""

REDUCE_PRIVATE_PROMPT = """You maintain JP's PRIVATE profile (sensitive: relationships, wellbeing,
support). GUARDRAILS: this is for attunement, NOT therapy. Distill to INSIGHT — durable patterns,
triggers, what helps, context. Do NOT hoard raw negative-thought dumps. Neutral, non-judgmental,
non-preachy. Merge the new facts into the current private profile; integrate, dedupe, keep concise.
Organize under: ## Relationships, ## Wellbeing & inner life, ## Support & frameworks,
## Patterns & what helps. Return the COMPLETE updated markdown only, no code fences.

===== CURRENT PRIVATE PROFILE =====
{profile}

===== NEW PRIVATE FACTS (bulk import) =====
{facts}"""


def _strip_fences(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip() + "\n"


def _backup(path):
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(path, encoding="utf-8") as s, open(f"{path}.bak.{stamp}", "w", encoding="utf-8") as d:
            d.write(s.read())


def do_reduce(key, state):
    pub = _load_facts(FACTS_PUB)
    priv = _load_facts(FACTS_PRIV)
    schema = open(SCHEMA, encoding="utf-8").read() if os.path.exists(SCHEMA) else ""

    if pub:
        cur = open(WIKI_PUBLIC, encoding="utf-8").read() if os.path.exists(WIKI_PUBLIC) else "# JP — profile\n"
        prompt = REDUCE_PUBLIC_PROMPT.format(schema=schema, profile=cur,
                                             facts="\n".join(f"- {x}" for x in pub))
        out = _strip_fences(gemini_call(key, prompt, max_tokens=8192))
        if len(out) > 100 and "##" in out:
            _backup(WIKI_PUBLIC)
            open(WIKI_PUBLIC, "w", encoding="utf-8").write(out)
            log(f"public profile merged ({len(pub)} facts → {len(out)} chars)")
        else:
            log("public reduce output suspicious — NOT written."); return

    if priv:
        os.makedirs(os.path.dirname(WIKI_PRIVATE), exist_ok=True)
        header = ("# JP — profile (PRIVATE layer) 🔒\n\n> SENSITIVE. Attunement, not therapy. "
                  "JP-only; never an outward surface. Distilled insight, not raw negativity.\n")
        cur = open(WIKI_PRIVATE, encoding="utf-8").read() if os.path.exists(WIKI_PRIVATE) else header
        prompt = REDUCE_PRIVATE_PROMPT.format(profile=cur, facts="\n".join(f"- {x}" for x in priv))
        out = _strip_fences(gemini_call(key, prompt, max_tokens=8192))
        if len(out) > 80:
            _backup(WIKI_PRIVATE)
            open(WIKI_PRIVATE, "w", encoding="utf-8").write(out)
            os.chmod(WIKI_PRIVATE, 0o600)
            log(f"private profile merged ({len(priv)} facts → {len(out)} chars)")
        else:
            log("private reduce output suspicious — NOT written."); return

    state["phase"] = "done"; save_state(state)
    log("REDUCE complete → done.")


# ---------- driver ----------
def cmd_demo():
    """Preview: extract in memory, run ONE map batch, print facts. No writes, no state change."""
    key = read_key()
    if not key:
        log("no GOOGLE_API_KEY — aborting."); return
    records = []
    extract_claude(records)
    extract_chatgpt(records)
    extract_gemini(records)
    batches = make_batches(records)
    from collections import Counter
    by_src = Counter(r["src"] for r in records)
    print(f"\n=== DEMO (no writes) ===")
    print(f"extracted {len(records)} conversations {dict(by_src)} → {len(batches)} batches "
          f"(model {MODEL}, ≠ voice)\n")
    if not batches:
        print("no records."); return
    batch = batches[0]
    print(f"processing batch 0 ({len(batch)} conversations)…\n")
    convos = "\n\n---\n\n".join(f"[{r['src']}] {r['title']}\n{r['text']}" for r in batch)
    prompt = MAP_PROMPT.format(guardrails=PRIVATE_GUARDRAILS, convos=convos)
    try:
        out = gemini_call(key, prompt)
    except Exception as e:
        print(f"Gemini error: {e}"); return
    pub, priv = parse_json_facts(out)
    print(f"--- PUBLIC facts ({len(pub)}) → would go to wiki/profile.md ---")
    for x in pub:
        print(f"  • {x}")
    print(f"\n--- PRIVATE facts ({len(priv)}) → would go to wiki/profile_private.md (JP-only) ---")
    for x in priv:
        print(f"  • {x}")
    print(f"\n(demo only — nothing written; full run would do ~{len(batches)} batches paced over ~1-2 days)")


def cmd_status():
    s = load_state()
    recs = sum(1 for _ in open(RECORDS, encoding="utf-8")) if os.path.exists(RECORDS) else 0
    batches = len(make_batches(load_records())) if recs else 0
    pub = sum(1 for _ in open(FACTS_PUB, encoding="utf-8")) if os.path.exists(FACTS_PUB) else 0
    priv = sum(1 for _ in open(FACTS_PRIV, encoding="utf-8")) if os.path.exists(FACTS_PRIV) else 0
    print(f"phase={s.get('phase')} map_batch={s.get('map_batch',0)}/{batches} "
          f"records={recs} public_facts={pub} private_facts={priv} model={MODEL}")


def cmd_step():
    key = read_key()
    if not key:
        log("no GOOGLE_API_KEY — aborting."); return
    s = load_state()
    phase = s.get("phase", "extract")
    if phase == "extract":
        do_extract(); s["phase"] = "map"; s["map_batch"] = 0; save_state(s)
        phase = "map"
    if phase == "map":
        do_map(key, s); s = load_state(); phase = s.get("phase")
    if phase == "reduce":
        do_reduce(key, s)
    elif phase == "done":
        log("bulk import already done — no-op.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "step"
    if cmd == "status":
        cmd_status()
    elif cmd == "demo":
        cmd_demo()
    elif cmd == "extract":
        do_extract()
        s = load_state(); s["phase"] = "map"; s.setdefault("map_batch", 0); save_state(s)
    elif cmd == "map":
        _key = read_key()
        if _key:
            _s = load_state(); do_map(_key, _s)
    elif cmd == "reduce":
        _key = read_key()
        if _key:
            _s = load_state(); do_reduce(_key, _s)
    else:
        cmd_step()
