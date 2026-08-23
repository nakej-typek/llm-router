"""Rychlý test availability rotace (bez čekání na reálný 429).

Opraveno 2026-08-14: volalo se `router.answer(q, ...)` s holým stringem, ale `answer()`
bere seznam zpráv — test padal na `'str' object has no attribute 'get'` hned v
`parse_override`, tedy dřív, než stihl cokoli otestovat. A cooldown se nastavoval na
`gemini/gemini-3.5-flash-lite`, což je jméno, které v poolu nefiguruje (kandidáti jdou
z `ROUTER_GEMINI_SMALL`/`_BIG`, dnes `*-latest` aliasy), takže i po opravě volání by
druhá část neměla co obejít a „rotace OK" by znamenalo jen že se nic nestalo.
Cooldown se proto bere z toho, co vrátilo první volání.

Běží živě proti API. Spouštět ručně, ne v CI.
"""
import time
from difficulty_router import EMBED_MODEL, build_index, FastEmbedEncoder
from availability import Availability
import router

enc = FastEmbedEncoder(name=EMBED_MODEL); enc.score_threshold = 0.0
index = build_index(enc)
avail = Availability()

msgs = [{"role": "user", "content": "jaké je hlavní město Španělska"}]

print("=== 1) normální volání ===")
tier, model, text, info = router.answer(msgs, enc, index, avail)
print("  model:", model)
print("  odpověď:", (text or "")[:70])

# `model` je zobrazovací štítek (nese i poznámku o rotaci), ne klíč cooldownu — ten je
# plné jméno kandidáta. Bereme je proto z routeru, ne z návratové hodnoty.
blocked = [router._GS, router._GB]
print(f"\n=== 2) {', '.join(blocked)} uměle na cooldownu → musí rotovat jinam ===")
for key in blocked:
    avail.cooldowns[key] = time.time() + 300
tier, model, text, info = router.answer(msgs, enc, index, avail)
print("  model:", model)
print("  odpověď:", (text or "")[:70])
hit_blocked = any(b.split("/")[-1] in model.split("[")[0] for b in blocked)
print("  ROTACE SELHALA (použit blokovaný model)" if hit_blocked else "  ROTACE OK")
