"""
Availability vrstva routeru — HYBRID (2026-07-29, po guardu na Archi).

Zdroj pravdy o dostupnosti = GUARD (active_pool.json), který drží živý pool. Router
už nepočítá kvóty sám — jen se ptá poolu. Co v poolu NENÍ (Gemini API, Opus zatím),
řeší lokální fallback (cooldown z 429). Každé volání se zapíše do usage.router.jsonl,
ze kterého guard počítá.

QUOTA_DIR: na Windows Syncthing (syncne se guardovi na Arch), na Archi lokální (env
AI_ROUTER_QUOTA_DIR). CLI/klíč kontroly zůstávají (přeskoč nenainstalovaný nástroj / bez klíče).
"""
import os
import re
import json
import time
import shutil
import pathlib

QUOTA_DIR = pathlib.Path(
    os.environ.get("AI_ROUTER_QUOTA_DIR")
    or pathlib.Path.home() / "syncthing" / "archlinux" / "ai_router" / "quota"
)
POOL_FILE = QUOTA_DIR / "active_pool.json"
LEDGER_FILE = QUOTA_DIR / "usage.router.jsonl"


def parse_retry_after(err):
    """Best-effort: vytáhni sekundy z 429 chyby (Gemini/LiteLLM texty se liší)."""
    s = str(err)
    for pat in (r'retry in ([\d.]+)s', r'retryDelay["\s:]+(\d+)s', r'Retry-After["\s:]+(\d+)'):
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def to_slug(model):
    """Router model string -> guard slug (provider/model)."""
    if model == "remote:codex":
        return "openai/gpt-free"
    if model.startswith("cli:claude:"):
        return "anthropic/" + model.split(":")[2]   # opus/sonnet/haiku/fable
    if model.startswith("gemini/"):
        return "google-gemini/" + model.split("/", 1)[1]
    if model.startswith("openrouter/"):
        return model  # už je "openrouter/<id>"
    return model


class Availability:
    def __init__(self):
        self.pool = self._load_pool()          # slug -> available(bool)
        self.cooldowns = {}                     # model -> until_ts (lokální belt+suspenders)

    def _load_pool(self):
        try:
            data = json.loads(POOL_FILE.read_text(encoding="utf-8"))
            return {p["slug"]: bool(p.get("available", True)) for p in data.get("pool", [])}
        except Exception:
            return {}

    def available(self, model):
        """(bool, důvod). Priorita: CLI/klíč check -> guard pool -> lokální cooldown."""
        now = time.time()
        if model.startswith("cli:"):
            tool = model.split(":")[1]
            if shutil.which(tool) is None:
                return False, "nenainstalováno"
        if model.startswith("openrouter/") and not os.environ.get("OPENROUTER_API_KEY"):
            return False, "chybí OpenRouter klíč"
        # GUARD pool (co v něm je, řídí on)
        slug = to_slug(model)
        if slug in self.pool:
            if not self.pool[slug]:
                return False, "guard: okno spotřebováno"
            return True, "guard:ok"
        # lokální fallback (Gemini/Opus zatím nejsou v poolu)
        cd = self.cooldowns.get(model)
        if cd and now < cd:
            return False, f"cooldown {int(cd - now)}s"
        return True, "ok"

    def _ledger(self, slug, event, resets_at=None):
        try:
            QUOTA_DIR.mkdir(parents=True, exist_ok=True)
            line = {"ts": time.time(), "slug": slug, "event": event}
            if event == "rate_limited":
                line["resets_at"] = resets_at
            with open(LEDGER_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass  # ledger je best-effort, nikdy neshodí volání

    def record_use(self, model):
        """Zapiš volání do ledgeru (guard z toho počítá)."""
        self._ledger(to_slug(model), "call")

    def record_429(self, model, err=None, default_wait=60):
        """Lokální cooldown (okamžitá rotace) + marker do ledgeru pro guard."""
        wait = (parse_retry_after(err) if err is not None else None) or default_wait
        self.cooldowns[model] = time.time() + wait
        self._ledger(to_slug(model), "rate_limited", resets_at=time.time() + wait)
        return wait
