"""Test konverzační paměti: druhý dotaz spoléhá na info z prvního.

POZOR na to, co se tu měří (opraveno 2026-08-14). Do 2026-08-14 se testovalo
"Jmenuju se Jan a mám nejradši Monero" → "jak se jmenuju a co mám nejradši".
Jenže `persona.py` vlepuje do každého volání `wiki/profile.md`, a ten obsahuje
JMÉNO i MONERO. Model tedy odpovídal správně, i kdyby se historie ztratila celá —
test nemohl selhat, takže nic netestoval. Proto je tu náhodný nonce: v profilu být
nemůže a v promptu druhého kola taky ne, takže jediná cesta, jak ho model může znát,
je zachovaný kontext konverzace.

Běží živě proti Claude CLI (fáze 2 = resume session). Spouštět ručně, ne v CI.
"""
import uuid

from difficulty_router import EMBED_MODEL, build_index, FastEmbedEncoder
from availability import Availability
from pool import curated_free
import router

enc = FastEmbedEncoder(name=EMBED_MODEL); enc.score_threshold = 0.0
index = build_index(enc); avail = Availability(); pool = curated_free()

NONCE = f"kod-{uuid.uuid4().hex[:8]}"

history = []


def turn(q):
    history.append({"role": "user", "content": q})
    tier, model, text, info = router.answer(history, enc, index, avail, extra_fallback=pool)
    history.append({"role": "assistant", "content": text or ""})
    print(f"\nTy: {q}")
    print(f"[{model} · {tier}] {(text or '').strip()[:160]}")
    return text or ""


turn(f"Zapamatuj si kódové slovo {NONCE}. Jen potvrď, nic víc.")
answer = turn("Jaké bylo to kódové slovo? Odpověz jen tím slovem.")

print(f"\nnonce={NONCE}")
print("PAMĚŤ DRŽÍ:" if NONCE in answer else "PAMĚŤ SELHALA:", NONCE in answer)
