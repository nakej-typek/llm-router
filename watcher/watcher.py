"""
Model-army watcher — V2.

Two daily jobs for the cost-aware AI router:

1) FREE POOL (unchanged from V1): fetch OpenRouter's public models API (no key),
   keep FREE text models (prompt+completion price = 0, output modality purely text),
   write free_models.json = the candidate pool the router draws from.

2) LIMITS MAP (new): the router needs the free-tier RATE LIMITS so it can self-account
   (count its own calls) and rotate before a bucket is spent — e.g. "GPT free ~10 uses /
   5h". Those numbers change irregularly, so we refresh daily into free_limits.json:
     - known_limits.json  = hand-maintained, high-confidence entries (JP edits). Wins.
     - community lists     = cheahjs/free-llm-api-resources + amardeeplakshkar/
                             awesome-free-llm-apis READMEs, distilled by FREE Gemini.
   Merged -> free_limits.json (manual overrides community on provider+model collision).

Schema is the one the router agreed on (W-021): a LIST keyed by provider+model
(glob ok, e.g. "*:free"). Standard windows use rpm/rpd/tpm; irregular windows (GPT
10/5h) use limit + window_sec.

Design split (W-020/W-021): the watcher OWNS this limits map. The router owns the
runtime availability layer — reads free_limits.json as the seed hint, self-counts for
fixed windows, and corrects from live 429 / Retry-After / X-RateLimit-* headers.

Article 1 / free only: OpenRouter models endpoint is public (no key); Gemini uses the
existing GOOGLE_API_KEY free tier (~1 call/day, negligible). stdlib ONLY (urllib) ->
no venv/pip. Best-effort: if Gemini or a list is unreachable, we keep the previous
community limits and never crash the pool fetch.
"""
import datetime
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
POOL_OUT = HERE / "free_models.json"
LIMITS_OUT = HERE / "free_limits.json"
KNOWN = HERE / "known_limits.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

# Community-maintained free-tier lists (raw markdown). Provider-level limits live
# near the top of each; we cap what we feed to Gemini to keep the call cheap.
COMMUNITY_LISTS = [
    "https://raw.githubusercontent.com/amardeeplakshkar/awesome-free-llm-apis/main/README.md",
    "https://raw.githubusercontent.com/cheahjs/free-llm-api-resources/main/README.md",
]
LIST_CHARS_CAP = 16000  # per list, chars fed to Gemini

# Gemini for the limits distillation. Daily ~1 call. Uses gemini-3.5-flash-lite:
# its own quota bucket, distinct from voice (gemini-3.1-flash-lite) and Hermes chat
# (gemini-flash-latest, which can already be 429). The bulk import that used this
# model is done, so the bucket is idle. 1 call/day can't dent it.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
HERMES_ENV = os.path.expanduser("~/.hermes/.env")

UA = {"User-Agent": "ai-router-watcher/2.0"}

# fields we accept on a limit entry (besides provider/model/notes)
_NUM_FIELDS = ("rpm", "rpd", "tpm", "tpd", "limit", "window_sec")


# ----------------------------- 1) FREE POOL -----------------------------

def _get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _is_zero(x):
    try:
        return float(x) == 0.0
    except (TypeError, ValueError):
        return False


def is_free(m):
    p = m.get("pricing", {}) or {}
    return _is_zero(p.get("prompt")) and _is_zero(p.get("completion"))


def outputs_text(m):
    """Keep only models that output PURE text (drop music/image: Lyria has
    output ['text','audio'], so 'text in outs' alone is not enough)."""
    arch = m.get("architecture", {}) or {}
    outs = arch.get("output_modalities")
    if isinstance(outs, list) and outs:
        return "text" in outs and "audio" not in outs and "image" not in outs
    modality = arch.get("modality", "")
    tail = modality.split("->")[-1] if "->" in modality else modality
    return "text" in tail and "audio" not in tail and "image" not in tail


def build_pool():
    data = _get_json(OPENROUTER_URL).get("data", [])
    free = []
    for m in data:
        if is_free(m) and outputs_text(m):
            arch = m.get("architecture", {}) or {}
            free.append({
                "id": m.get("id"),
                "name": m.get("name"),
                "context": m.get("context_length"),
                "inputs": arch.get("input_modalities") or arch.get("modality"),
            })
    free.sort(key=lambda x: -(x["context"] or 0))
    out = {
        "fetched_at": datetime.date.today().isoformat(),
        "source": OPENROUTER_URL,
        "total_models_seen": len(data),
        "free_count": len(free),
        "free_models": free,
    }
    POOL_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pool]   {len(data)} models seen -> {len(free)} FREE -> {POOL_OUT.name}")
    for m in free[:12]:
        print(f"           {(m['id'] or '')[:46]:<48} ctx={m['context'] or 0:>9,}")
    if len(free) > 12:
        print(f"           ... and {len(free) - 12} more")
    return out


# ----------------------------- 2) LIMITS MAP ----------------------------

def read_google_key():
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


def fetch_text(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def _clean_entry(v, source):
    """Validate/normalize one limit dict into the agreed schema, or None."""
    if not isinstance(v, dict):
        return None
    provider = str(v.get("provider", "")).strip().lower()
    if not provider:
        return None
    entry = {"provider": provider,
             "model": str(v.get("model", "*")).strip() or "*"}
    for f in _NUM_FIELDS:
        if v.get(f) is None:
            continue
        try:
            entry[f] = int(v[f])
        except (TypeError, ValueError):
            pass
    entry["notes"] = str(v.get("notes", ""))[:140]
    entry["source"] = source
    return entry


LIMITS_PROMPT = (
    "You are given README text from community-maintained lists of FREE LLM API tiers. "
    "Extract the free-tier rate limits per provider (and per model where the list is "
    "model-specific). Return STRICT JSON ONLY, no prose, no code fences, exactly:\n"
    '{"limits": [{"provider": "<slug>", "model": "<model-or-*>", "rpm": <int|null>, '
    '"rpd": <int|null>, "tpm": <int|null>, "notes": "<short>"}]}\n'
    "Rules:\n"
    "- provider = lowercase kebab slug (groq, cerebras, mistral, cohere, google-gemini, "
    "nvidia, openrouter, ...). model = specific id, or '*' if the limit is account-wide, "
    "or '*:free' style glob if that's how the list frames it.\n"
    "- rpm = requests/min, rpd = requests/day, tpm = tokens/min. Convert any 'RPM'->rpm, "
    "'RPD'->rpd, 'TPM'->tpm. Use null for anything not stated. Integers only (no commas/ranges; "
    "if a range, take the lower bound).\n"
    "- ONLY genuinely free tiers with NO credit card / NO trial credits. Skip paid, skip "
    "anything uncertain. Better to omit than to guess. Keep notes under ~12 words.\n\n"
    "README TEXT:\n"
)


def gemini_extract_limits(key, corpus):
    body = {
        "contents": [{"parts": [{"text": LIMITS_PROMPT + corpus}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 4096,
                             "responseMimeType": "application/json"},
    }
    url = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateContent?key={key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={**UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        resp = json.load(r)
    text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].lstrip("json").strip()
    parsed = json.loads(text)
    rows = parsed.get("limits", parsed) if isinstance(parsed, dict) else parsed
    out = []
    for v in rows or []:
        e = _clean_entry(v, "community")
        if e:
            out.append(e)
    return out


def load_known():
    try:
        data = json.loads(KNOWN.read_text(encoding="utf-8"))
        out = []
        for v in data.get("limits", []) or []:
            e = _clean_entry(v, "manual")
            if e:
                out.append(e)
        return out
    except (OSError, ValueError) as e:
        print(f"[limits] known_limits.json unreadable ({e}); continuing without overrides")
        return []


def _key(e):
    return (e["provider"], e["model"])


def build_limits():
    known = load_known()
    community = []
    key = read_google_key()
    if not key:
        print("[limits] no GOOGLE_API_KEY -> skipping community research, using known_limits only")
    else:
        parts = []
        for url in COMMUNITY_LISTS:
            try:
                parts.append(f"### SOURCE: {url}\n" + fetch_text(url)[:LIST_CHARS_CAP])
            except (urllib.error.URLError, OSError) as e:
                print(f"[limits] could not fetch {url}: {e}")
        if parts:
            corpus = "\n\n".join(parts)
            for attempt in range(1, 4):
                try:
                    community = gemini_extract_limits(key, corpus)
                    if community:
                        print(f"[limits] Gemini extracted {len(community)} community "
                              f"entries (attempt {attempt})")
                        break
                    print(f"[limits] attempt {attempt}: Gemini returned 0 entries, retrying")
                except (urllib.error.HTTPError, urllib.error.URLError, KeyError,
                        IndexError, ValueError, OSError) as e:
                    print(f"[limits] attempt {attempt}: Gemini extract failed ({e})")
                if attempt < 3:
                    time.sleep(5)
            if not community:
                print("[limits] all attempts empty/failed; keeping manual + prior community")

    # if research produced nothing, reuse prior community entries so we don't lose the map
    if not community and LIMITS_OUT.exists():
        try:
            prior = json.loads(LIMITS_OUT.read_text(encoding="utf-8")).get("limits", [])
            community = [e for e in prior if isinstance(e, dict)
                         and e.get("source") == "community"]
            if community:
                print(f"[limits] preserved {len(community)} prior community entries")
        except (OSError, ValueError):
            pass

    # merge: community first, manual overrides on (provider, model)
    merged = {_key(e): e for e in community if _clean_entry(e, e.get("source", "community"))}
    for e in known:
        merged[_key(e)] = e
    rows = sorted(merged.values(), key=lambda e: (e["provider"], e["model"]))

    out = {
        "fetched_at": datetime.date.today().isoformat(),
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema": ("router keys on provider+model (glob ok). standard windows: rpm/rpd/tpm; "
                   "irregular windows: limit + window_sec. self-accounting: "
                   "available = count(own calls in window) < limit."),
        "sources": {"manual": KNOWN.name, "community": COMMUNITY_LISTS,
                    "gemini_model": GEMINI_MODEL},
        "limits": rows,
    }
    LIMITS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    n_manual = sum(1 for e in rows if e["source"] == "manual")
    print(f"[limits] {len(rows)} entries ({n_manual} manual, {len(rows)-n_manual} community) "
          f"-> {LIMITS_OUT.name}")
    for e in rows:
        w = " ".join(f"{f}={e[f]}" for f in _NUM_FIELDS if f in e)
        print(f"           {e['provider']}/{e['model']:<18} {w:<34} [{e['source']}]")
    return out


def main():
    print("=" * 74)
    try:
        build_pool()
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[pool]   FAILED: {e} (keeping prior free_models.json)")
    print("-" * 74)
    build_limits()
    print("=" * 74)


if __name__ == "__main__":
    main()
