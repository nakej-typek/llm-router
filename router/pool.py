"""
Kurátor free poolu — z toho, co watcher najde (free_models.json, klidně 15-400 modelů),
vybere PRAVIDLEM jen pár, co za pozornost stojí. Ne ruční seznam, ať to funguje i na
budoucí pool. Vrací je jako 'openrouter/<id>' pro LiteLLM.

Filtr:
  - vyhoď ne-chat / meta modely (content-safety, guard, openrouter/free…)
  - vyžaduj rozumnou velikost (>= MIN_PARAMS_B mld. parametrů, parsuje se z názvu)
  - seřaď podle velikosti (pak kontextu), nech top N
"""
import re
import json
import pathlib

HERE = pathlib.Path(__file__).parent
POOL_FILE = HERE / "free_models.json"

EXCLUDE_SUBSTR = ["content-safety", "guard", "moderation", "openrouter/free"]
MIN_PARAMS_B = 20


def _params_b(model_id):
    """Celkové parametry v miliardách z názvu (první '<num>b'); 0 když neuvedeno."""
    m = re.search(r"(\d+)\s*b\b", model_id.lower())
    return int(m.group(1)) if m else 0


def curated_free(limit=5):
    if not POOL_FILE.exists():
        return []
    try:
        data = json.loads(POOL_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    scored = []
    for m in data.get("free_models", []):
        mid = m.get("id", "")
        if any(x in mid for x in EXCLUDE_SUBSTR):
            continue
        p = _params_b(mid)
        if p < MIN_PARAMS_B:
            continue
        scored.append((p, m.get("context") or 0, mid))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [f"openrouter/{mid}" for _, _, mid in scored[:limit]]


if __name__ == "__main__":
    picks = curated_free()
    print(f"Kurátor vybral {len(picks)} z free poolu (filtr: chat, >= {MIN_PARAMS_B}B, top by velikost):")
    for p in picks:
        print("  ", p)
