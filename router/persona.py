"""
Persona — system prompt routeru: (1) PROFIL JP z wiki (aby tě znal), (2) ARCHITEKTURA
routeru (aby pravdivě popsal sám sebe), (3) RUNTIME blok tohoto dotazu (vybraný model,
stupeň, přeskočení) — injektovaný pro VÍTĚZE, ne kandidáta.

INVARIANT: router jen ČTE výstup učení (wiki); destiluje Gemini/Hermes. Jen PUBLIC profil.
ENV: LEARNING_WIKI (jinak auto-find), AI_ROUTER_HOST_LABEL.
"""
import os
import pathlib


def _find_wiki():
    env = os.environ.get("LEARNING_WIKI")
    if env:
        return pathlib.Path(env)
    for c in (pathlib.Path.home() / "syncthing" / "archlinux" / "ai_router"
              / "learning_core" / "wiki" / "profile.md",):
        if c.exists():
            return c
    return pathlib.Path.home() / "syncthing" / "archlinux" / "ai_router" / \
        "learning_core" / "wiki" / "profile.md"


WIKI = _find_wiki()
MAX_CHARS = 3000  # zkráceno (JP 2026-07-30): 12000 = zbytečná daň za zpracování na KAŽDÝ dotaz

PROFILE_PREAMBLE = (
    "Jsi osobní AI asistent JP (Jan). Níže je, co o něm víš z jeho profilu — využij to pro "
    "relevantní a osobní odpovědi, ale neopakuj profil zpět, když se k dotazu nehodí. Piš česky, "
    "anglické technické termíny nech anglicky. Buď přímý a upřímný, bez zbytečné vaty.\n\n"
    "=== PROFIL JP ===\n{profile}\n=== konec profilu ==="
)

ARCHITECTURE = (
    "=== O TOBĚ (krátce) ===\n"
    "Jsi „ai-router\" — cost-aware router před víc modely (Gemini/GPT/Claude/OpenRouter free), "
    "co si JP postavil, běžící jako persistentní služba na jeho Arch laptopu. Vždy použij "
    "RUNTIME info níže pro skutečný model/stupeň TOHOTO dotazu — ne domněnku.\n"
    "Když se JP zeptá na PODROBNOU architekturu (jak přesně fungueš, proč je něco pomalé, "
    "co běží jako služba…), NEHÁDEJ a NEHALUCINUJ — přečti si přes svůj souborový nástroj "
    "(read_file) soubor learning_core/wiki/architecture.md (v ~/syncthing/archlinux/ai_router/), "
    "kde je to popsané fakticky a aktuálně, a odpověz z toho."
)

_cache = {"mtime": None, "text": ""}


def _profile_text():
    try:
        if not WIKI.exists():
            return ""
        mt = WIKI.stat().st_mtime
        if _cache["mtime"] != mt:
            _cache["mtime"] = mt
            _cache["text"] = WIKI.read_text(encoding="utf-8", errors="replace").strip()[:MAX_CHARS]
        return _cache["text"]
    except Exception:
        return ""


def _runtime_block(tier, model_label, skipped, host):
    lines = ["=== RUNTIME (tento konkrétní dotaz) ==="]
    if model_label:
        lines.append(f"- Právě tě zpracovává model: {model_label}")
    if tier:
        lines.append(f"- Klasifikace obtížnosti: {tier}")
    if host:
        lines.append(f"- Host: {host}")
    if skipped:
        lines.append("- Přeskočené (nedostupné) modely: " + "; ".join(skipped))
    return "\n".join(lines)


def build_system(tier=None, model_label=None, skipped=None, host=None):
    """System message: profil + architektura (+ runtime, když je znám vítěz)."""
    profile = _profile_text()
    parts = [PROFILE_PREAMBLE.format(profile=profile) if profile
             else "Jsi osobní AI asistent JP (Jan). Piš česky, přímo a bez vaty."]
    parts.append(ARCHITECTURE)
    if model_label or tier:
        parts.append(_runtime_block(tier, model_label, skipped, host))
    return {"role": "system", "content": "\n\n".join(parts)}


def system_message():
    """Zpětná kompatibilita — profil + architektura bez runtime bloku."""
    return build_system()
