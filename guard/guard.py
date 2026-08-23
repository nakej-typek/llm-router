#!/usr/bin/env python3
"""
Availability GUARD for the AI router (W-023, JP's steer).

Three-layer split:
  1. WATCHER (daily)     edits the RULES        -> watcher/free_limits.json (+ free_models.json,
                                                   optional free_ranked.json)
  2. GUARD (this, always-on) reads rules + our OWN usage -> quota/active_pool.json
  3. ROUTER              reads active_pool.json ONLY -> picks the best AVAILABLE model, never
                                                        does quota math.

The guard drops a model from availability when its window is spent and auto-restores it when the
window rolls. It NEVER polls any provider — it self-accounts from our own call ledger (free, exact).

LEDGER (W-023 one-writer-per-file, so Syncthing never makes conflict copies):
  quota/usage.gpt.jsonl     <- Arch gpt_endpoint appends (GPT/Codex calls)
  quota/usage.router.jsonl  <- Windows router appends (its own Gemini/OpenRouter free calls)
Each line: {"ts": <epoch>, "slug": "<provider>/<model>", "event": "call"|"rate_limited",
            "resets_at": <epoch optional>}. The guard reads the UNION.

BUCKETS: a limit rule in free_limits.json (e.g. openrouter/*:free rpm20 rpd50) is a shared BUCKET —
every model matching it draws the same quota. The guard maps each candidate model AND each ledger
entry to its bucket via glob match on "provider/model", then counts uses per bucket per window.

Article 1: reads only local/synced files, no network, no keys. stdlib only.
"""
import datetime
import fnmatch
import json
import os
import pathlib
import time

AIR = pathlib.Path(os.path.expanduser("~/syncthing/archlinux/ai_router"))
WATCHER = AIR / "watcher"
# QUOTA dir is env-configurable so the single-box move (W-024) is a config flip, not a
# code change: once the router co-locates on Arch, point QUOTA_DIR at a LOCAL (non-synced)
# path so the hot-path ledger/active_pool.json never touch Syncthing.
QUOTA = pathlib.Path(os.path.expanduser(
    os.environ.get("AI_ROUTER_QUOTA_DIR")
    or os.environ.get("QUOTA_DIR")
    or str(AIR / "quota")))
GUARD = AIR / "guard"

FREE_MODELS = WATCHER / "free_models.json"
FREE_LIMITS = WATCHER / "free_limits.json"
FREE_RANKED = WATCHER / "free_ranked.json"      # optional; produced by watcher quality-rank
INTERFACE = GUARD / "interface_models.json"
LEDGERS = [QUOTA / "usage.gpt.jsonl", QUOTA / "usage.router.jsonl"]
ACTIVE_POOL = QUOTA / "active_pool.json"

INTERVAL = int(os.environ.get("GUARD_INTERVAL", "60"))  # seconds between recomputes
# token-rate fields we do NOT gate on (we count requests, not tokens); kept as info only
_TOKEN_FIELDS = ("tpm", "tpd")


def _load_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def load_rules():
    """Return list of buckets: {pattern, provider, model, windows:[(limit,window_sec,label)]}."""
    data = _load_json(FREE_LIMITS, {})
    rules = []
    for e in data.get("limits", []) or []:
        provider = str(e.get("provider", "")).strip()
        model = str(e.get("model", "*")).strip() or "*"
        if not provider:
            continue
        windows = []
        if e.get("rpm") is not None:
            windows.append((int(e["rpm"]), 60, "rpm"))
        if e.get("rpd") is not None:
            windows.append((int(e["rpd"]), 86400, "rpd"))
        if e.get("limit") is not None and e.get("window_sec"):
            windows.append((int(e["limit"]), int(e["window_sec"]), "window"))
        if not windows:
            continue  # token-only or empty rule: not request-gateable
        rules.append({"pattern": f"{provider}/{model}", "provider": provider,
                      "model": model, "windows": windows})
    return rules


def _specificity(pattern):
    # more specific = fewer wildcards, then longer literal
    return (-pattern.count("*"), len(pattern.replace("*", "")))


def match_bucket(slug, rules):
    """Best (most specific) rule whose 'provider/model' glob matches slug, or None."""
    best = None
    for r in rules:
        if fnmatch.fnmatch(slug, r["pattern"]):
            if best is None or _specificity(r["pattern"]) > _specificity(best["pattern"]):
                best = r
    return best


def load_candidates():
    """Pool = OpenRouter free models (from watcher) + interface models (GPT, ...)."""
    cands = []
    fm = _load_json(FREE_MODELS, {})
    for m in fm.get("free_models", []) or []:
        mid = m.get("id")
        if not mid:
            continue
        cands.append({"slug": f"openrouter/{mid}", "provider": "openrouter",
                      "model": mid, "kind": "api-free", "ctx": m.get("context") or 0})
    im = _load_json(INTERFACE, {})
    for m in im.get("models", []) or []:
        if m.get("slug"):
            cands.append({"slug": m["slug"], "provider": m.get("provider", ""),
                          "model": m.get("model", ""), "kind": m.get("kind", "interface"),
                          "ctx": m.get("ctx") or 0})
    return cands


def load_ledger(rules, now):
    """Read union of ledgers -> per-bucket-pattern: list of call ts + latest rate_limited."""
    calls = {}          # pattern -> [ts, ...]
    rl = {}             # pattern -> {"ts":, "resets_at":}
    for path in LEDGERS:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            slug = e.get("slug")
            ts = e.get("ts")
            if not slug or not isinstance(ts, (int, float)):
                continue
            b = match_bucket(slug, rules)
            if not b:
                continue
            pat = b["pattern"]
            ev = e.get("event", "call")
            if ev == "rate_limited":
                cur = rl.get(pat)
                if cur is None or ts > cur["ts"]:
                    rl[pat] = {"ts": ts, "resets_at": e.get("resets_at")}
            else:
                calls.setdefault(pat, []).append(ts)
    return calls, rl


def evaluate(cand, rules, calls, rl, now):
    bucket = match_bucket(cand["slug"], rules)
    if not bucket:
        return {"available": True, "rank_ctx": cand.get("ctx", 0),
                "windows": [], "resets_at": None, "bucket": None}
    pat = bucket["pattern"]
    ts_list = calls.get(pat, [])
    wins, resets = [], []
    available = True
    for limit, wsec, label in bucket["windows"]:
        used = sum(1 for t in ts_list if t >= now - wsec)
        remaining = max(0, limit - used)
        win_avail = remaining > 0
        w_reset = None
        if not win_avail:
            in_win = sorted(t for t in ts_list if t >= now - wsec)
            w_reset = (in_win[0] + wsec) if in_win else (now + wsec)
        wins.append({"label": label, "limit": limit, "window_sec": wsec,
                     "used": used, "remaining": remaining,
                     "resets_at": round(w_reset, 1) if w_reset else None})
        if not win_avail:
            available = False
            resets.append(w_reset)
    # authoritative live rate_limited marker overrides self-count
    marker = rl.get(pat)
    if marker and marker["ts"] >= now - max(w[1] for w in bucket["windows"]):
        r_at = marker.get("resets_at")
        # honor marker only if it hasn't already elapsed
        if r_at is None or r_at > now:
            available = False
            resets.append(r_at if r_at else now + bucket["windows"][-1][1])
    resets_at = max([r for r in resets if r], default=None)
    return {"available": available, "rank_ctx": cand.get("ctx", 0),
            "windows": wins, "resets_at": round(resets_at, 1) if resets_at else None,
            "bucket": pat}


def compute_pool():
    rules = load_rules()
    cands = load_candidates()
    ranked = _load_json(FREE_RANKED, {})
    rankmap = {}
    if isinstance(ranked, dict):
        for k, v in (ranked.get("ranks", ranked) or {}).items():
            try:
                rankmap[k] = float(v if not isinstance(v, dict) else v.get("rank"))
            except (TypeError, ValueError):
                pass
    now = time.time()
    calls, rl = load_ledger(rules, now)
    pool = []
    for c in cands:
        ev = evaluate(c, rules, calls, rl, now)
        rank = rankmap.get(c["slug"])
        pool.append({
            "slug": c["slug"], "provider": c["provider"], "model": c["model"],
            "kind": c["kind"], "available": ev["available"],
            "rank": rank,                      # None until watcher quality-rank lands
            "resets_at": ev["resets_at"],
            "bucket": ev["bucket"], "windows": ev["windows"],
        })
    # order: available first, then by quality rank (desc) if known, else ctx (desc) interim
    pool.sort(key=lambda p: (
        0 if p["available"] else 1,
        -(p["rank"] if p["rank"] is not None else -1),
        -next((c["ctx"] for c in cands if c["slug"] == p["slug"]), 0),
    ))
    out = {
        "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": ("router reads this ONLY: pick the best AVAILABLE entry that clears the task's "
                 "quality bar; on a live call failure rotate to the next available. "
                 "rank=None means watcher quality-rank not deployed yet (interim order = ctx)."),
        "counts": {"total": len(pool),
                   "available": sum(1 for p in pool if p["available"]),
                   "gated": sum(1 for p in pool if p["bucket"])},
        "pool": pool,
    }
    QUOTA.mkdir(parents=True, exist_ok=True)
    tmp = ACTIVE_POOL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(ACTIVE_POOL)  # atomic swap so the router never reads a half-written file
    return out


def main():
    once = os.environ.get("GUARD_ONCE") == "1"
    while True:
        try:
            out = compute_pool()
            gpt = next((p for p in out["pool"] if p["slug"] == "openai/gpt-free"), None)
            gpt_s = (f"gpt avail={gpt['available']} resets_at={gpt['resets_at']}"
                     if gpt else "gpt n/a")
            print(f"[guard] pool={out['counts']['available']}/{out['counts']['total']} "
                  f"available, gated={out['counts']['gated']} | {gpt_s}")
        except Exception as e:  # never let the daemon die on a transient read
            print(f"[guard] recompute error: {e}")
        if once:
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
