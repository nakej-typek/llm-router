#!/usr/bin/env python3
"""Vyřízne z raw korpusu sekce, které nevygeneroval člověk, ale router sám.

PROČ (změřeno 2026-08-14): Claude Agent SDK běží s CWD nasazeného routeru, takže si
každé směrování na cli:claude:* zakládá vlastní Claude Code session v
~/.claude/projects/-home-user--local-share-ai-router-router/. cc_ingest je bral
jako konverzace JP → 132 598 znaků (11,7 %) `claude-code-sessions.arch.md` je provoz
routeru, zapsaný pod štítkem "JP:", s plochou historií zopakovanou v každém tahu.
Driver z toho psal skilly o učebnicovém HTTP místo o JP.

Budoucí zápisy zastaví EXCLUDE v cc_ingest.py. Tenhle skript uklidí, co už tam je.

OFFSETY SE PŘEPOČÍTÁVAJÍ, NEMAŽOU. Driver i distiller si drží pozice v bajtech; kdyby
se soubor jen zkrátil, ukazovaly by doprostřed cizího textu. Pro každý uložený offset se
spočítá, kolik vyříznutých bajtů mu předcházelo, a o tolik se posune. Kdo měl přečteno
do X, má po úklidu přečteno do X - (smazáno před X) — tedy pořád totéž místo v textu.

Seznam `chunks` ve stavu driveru se ZÁMĚRNĚ nechává, jak je. Slouží jen k počítání,
kolikátý je to pokus o týž (soubor, start), a z toho se skládá session id. Zastaralý
záznam tedy nejhůř způsobí, že nový pokus dostane id s `-a1` místo bez přípony —
neškodné. Migrovat ho napůl by bylo horší než ho nemigrovat vůbec.

Použití:
    python3 purge_router_sessions.py            # NÁHLED, nic nemění
    python3 purge_router_sessions.py --apply    # provede (dělá zálohy)
"""
import json
import os
import re
import shutil
import sys
import time

BASE = os.path.expanduser("~/syncthing/archlinux/ai_router/learning_core")
RAW = os.path.join(BASE, "raw")
DISTILLER_STATE = os.path.join(BASE, ".distiller_state.json")
DRIVER_STATE = os.path.expanduser("~/.local/share/ai_router/driver/driver_state.json")

# Hlavička sekce, kterou píše cc_ingest: "## <cesta> · <8 hex> · <stroj> · <YYYY-MM-DD HH:MM>"
# Musí sedět CELÁ — uvnitř konverzací jsou běžné markdown nadpisy ("## Stav") a ty
# sekce nedělí. Na tomhle rozdílu stojí, jestli se vyřízne přesně to, co má.
HEADER = re.compile(
    r"^## (?P<path>.+?) · [0-9a-f]{8} · \w+ · \d{4}-\d{2}-\d{2} \d{2}:\d{2}$", re.M)

# Co je "ne-lidský" zdroj. Stejné kritérium jako EXCLUDE v cc_ingest.py, ale nad CESTOU
# (v hlavičce jsou pomlčky slugu už převedené na lomítka).
BAD = ("/local/share/ai/router", "/local/share/ai_router")


def spans(text):
    """[(start, end, path)] pro každou sekci."""
    heads = list(HEADER.finditer(text))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out.append((m.start(), end, m.group("path")))
    return out


def plan(text):
    """([(start, end)] k vyříznutí, kolik znaků, kolik sekcí)."""
    cuts = [(s, e) for s, e, p in spans(text) if any(b in p for b in BAD)]
    return cuts, sum(e - s for s, e in cuts), len(cuts)


def remap(offset, cuts):
    """Nový offset po vyříznutí. Offset uvnitř vyříznutého úseku spadne na jeho začátek."""
    removed = 0
    for s, e in cuts:
        if offset >= e:
            removed += e - s
        elif offset > s:
            removed += offset - s
    return offset - removed


def main():
    apply = "--apply" in sys.argv
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report = []

    for name in sorted(os.listdir(RAW)):
        if not name.endswith(".md") or name.lower() == "readme.md":
            continue
        path = os.path.join(RAW, name)
        text = open(path, encoding="utf-8", errors="replace").read()
        cuts, n_chars, n_secs = plan(text)
        if not cuts:
            continue
        report.append((name, n_chars, n_secs, len(text)))
        if not apply:
            continue

        shutil.copy2(path, f"{path}.bak.purge.{stamp}")
        keep, prev = [], 0
        for s, e in cuts:
            keep.append(text[prev:s])
            prev = e
        keep.append(text[prev:])
        new = "".join(keep)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new)
        os.replace(tmp, path)

        # --- offsety ---
        for state_path, getter in ((DISTILLER_STATE, "distiller"), (DRIVER_STATE, "driver")):
            if not os.path.exists(state_path):
                continue
            st = json.load(open(state_path, encoding="utf-8"))
            shutil.copy2(state_path, f"{state_path}.bak.purge.{stamp}")
            changed = False
            if getter == "distiller":
                if name in st:
                    old = st[name]
                    st[name] = remap(old, cuts)
                    print(f"    distiller {name}: {old} -> {st[name]}")
                    changed = True
            else:
                f = (st.get("files") or {}).get(name)
                if f:
                    for k in ("tail_start", "backfill_offset", "tail_offset"):
                        if k in f:
                            old = f[k]
                            f[k] = remap(old, cuts)
                            print(f"    driver {name}.{k}: {old} -> {f[k]}")
                    changed = True
            if changed:
                tmp = state_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(st, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, state_path)

    if not report:
        print("nic k vyříznutí — korpus je čistý.")
        return 0
    print("NÁHLED" if not apply else "PROVEDENO")
    for name, n_chars, n_secs, total in report:
        print(f"  {name}: {n_secs} sekcí, {n_chars} znaků "
              f"({n_chars / total * 100:.1f} % souboru)")
    if not apply:
        print("\nnic se nezměnilo. Spusť s --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
