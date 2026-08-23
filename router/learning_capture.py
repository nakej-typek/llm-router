"""
Learning capture — router jako trychtýř učení (R-001).

Každá konverzace přes router se přidá do raw feedu, který Hermesův Gemini destiler
zpracuje do wiki. INVARIANT: tady se NEdestiluje — jen se zapisuje syrový materiál;
učení dělá Gemini (Hermes). Best-effort: chyba zápisu NIKDY neshodí odpověď.

Per-STROJ soubor (Win+Arch se nezraní přes Syncthing). ENV: LEARNING_RAW_DIR, CC_MACHINE.
"""
import os
import time
import pathlib


def _find_raw_dir():
    env = os.environ.get("LEARNING_RAW_DIR")
    if env:
        return pathlib.Path(env)
    candidates = [
        pathlib.Path.home() / "syncthing" / "archlinux" / "ai_router" / "learning_core" / "raw",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


RAW_DIR = _find_raw_dir()
MACHINE = os.environ.get("CC_MACHINE") or os.environ.get("COMPUTERNAME") or "unknown"
OUT = RAW_DIR / f"router-conversations.{MACHINE.lower()}.md"
ASSISTANT_TRUNC = 600


def capture(user_text, assistant_text):
    try:
        u = (user_text or "").strip()
        a = (assistant_text or "").strip()
        if len(u) < 3 or not a:
            return
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(f"\n## router · {MACHINE} · {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"JP: {u}\n")
            f.write(f"Asistent: {a[:ASSISTANT_TRUNC]}\n")
    except Exception:
        pass  # best-effort — nikdy neshodí odpověď
