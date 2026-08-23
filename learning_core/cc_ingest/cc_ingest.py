"""
Claude Code session ingester — sype JP konverzace do learning raw feedu.

Čte ~/.claude/projects/<proj>/<session>.jsonl, drží per-soubor offset (jen NOVÉ řádky),
vytáhne JP user zprávy (+ zkrácenou asistentovu odpověď pro kontext), přidá čistý markdown
do raw feedu, který Gemini destiler zpracuje. Vyloučí citlivé projekty (therapy).

Běží na každém stroji (Win + Arch). INVARIANT: tady NEJSOU žádná model volání — jen
extrakce; destilaci dělá Gemini (Hermes), ne tenhle skript.

ENV: CC_PROJECTS_DIR, CC_RAW_OUT, CC_STATE, CC_MACHINE.
"""
import os
import json
import glob
import time
import pathlib

PROJECTS_DIR = pathlib.Path(os.environ.get("CC_PROJECTS_DIR")
                            or (pathlib.Path.home() / ".claude" / "projects"))
# Slug substrings to skip. Dva různé důvody, proto ten komentář u každého:
#   "therapy"  — soukromí (terapeutický bot, do korpusu nepatří).
#   "ai-router-router" / "ai_router" pod ~/.local/share — TOHLE NENÍ KONVERZACE S JP.
#     Zjištěno 2026-08-14 měřením: 132 598 znaků, tj. 11,7 % `claude-code-sessions.arch.md`,
#     pochází z adresáře ~/.claude/projects/-home-user--local-share-ai-router-router/.
#     Claude Agent SDK totiž běží s CWD nasazeného routeru, takže si KAŽDÉ směrování na
#     cli:claude:* zakládá vlastní Claude Code session — a ta se sem ingestovala jako by ji
#     psal JP. Tři důsledky, všechny špatné:
#       1) duplicita: totéž už zachytává learning_capture do router-conversations.*.md;
#       2) kvadratické nafouknutí: router posílá plochou historii, takže tah N obsahuje
#          tahy 1..N-1 znovu (proto 307 výskytů jednoho testovacího tématu);
#       3) záměna mluvčího: prompt routeru ("Uživatel: …") se zapsal pod štítkem "JP:".
#     Driver z toho pak psal skilly o učebnicovém HTTP místo o JP.
EXCLUDE = ["therapy", "local-share-ai-router", "local-share-ai_router"]
ASSISTANT_TRUNC = 400
MIN_USER_LEN = 3
MACHINE = os.environ.get("CC_MACHINE") or os.environ.get("COMPUTERNAME") or "unknown"

# Per-STROJ raw soubor (Win i Arch píšou svůj -> žádný Syncthing konflikt, jako u ledgeru).
OUT = pathlib.Path(os.environ.get("CC_RAW_OUT")
                   or (pathlib.Path.home() / "syncthing" / "archlinux" / "ai_router"
                       / "learning_core" / "raw"
                       / f"claude-code-sessions.{MACHINE.lower()}.md"))
# STATE je LOKÁLNÍ (ne do Syncthingu — jinak by se offsety mezi stroji přepisovaly).
STATE = pathlib.Path(os.environ.get("CC_STATE")
                     or (pathlib.Path(__file__).parent / ".cc_ingest_state.json"))


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _clean_user(c):
    """Vrať skutečnou JP zprávu, nebo None když je to harness šum."""
    t = c.strip()
    # voice dikce začíná "speaking: " (i dvakrát) — nech obsah, sundej sentinel
    while t.lower().startswith("speaking:"):
        t = t[len("speaking:"):].strip()
    if len(t) < MIN_USER_LEN:
        return None
    # harness injekce (slash-commandy, system-reminder, caveaty, tool výstupy) = ne JP
    if t.startswith("<") or t.startswith(("Caveat:", "[Request interrupted",
                                          "Result of calling", "Tool ran")):
        return None
    return t


def extract_turns(lines):
    turns = []
    for line in lines:
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        msg = o.get("message") or {}
        if t == "user":
            c = msg.get("content")
            if isinstance(c, str):
                clean = _clean_user(c)
                if clean:
                    turns.append(("JP", clean))
        elif t == "assistant":
            c = msg.get("content")
            if isinstance(c, list):
                txt = " ".join(b.get("text", "") for b in c
                               if isinstance(b, dict) and b.get("type") == "text").strip()
                if txt:
                    turns.append(("Asistent", txt[:ASSISTANT_TRUNC]))
    return turns


def main():
    state = load_state()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total_new, sessions_touched = 0, 0
    for path in sorted(glob.glob(str(PROJECTS_DIR / "*" / "*.jsonl"))):
        p = pathlib.Path(path)
        slug = p.parent.name
        if any(x in slug.lower() for x in EXCLUDE):
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        done = state.get(path, 0)
        if len(lines) <= done:
            continue
        turns = extract_turns(lines[done:])
        state[path] = len(lines)
        if not turns:
            continue
        proj = slug.replace("C--Users-JP-", "").replace("-", "/")
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(f"\n## {proj} · {p.stem[:8]} · {MACHINE} · {time.strftime('%Y-%m-%d %H:%M')}\n")
            for who, text in turns:
                f.write(f"{who}: {text}\n")
        total_new += len(turns)
        sessions_touched += 1
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ingested {total_new} new turns from {sessions_touched} sessions -> {OUT}")


if __name__ == "__main__":
    main()
