"""
V1 difficulty router — Stage 1 klasifikátor.
Roztřídí dotaz do 3 fixních stupňů (triviální / střední / těžký) podle OBSAHU,
multilingvně (CZ + EN v jednom prostoru), lokálně, bez LLM volání.

Design: viz REQUIREMENTS "V1 PRINCIP ROUTERU".
- model-agnostický (nezná jména modelů) -> škáluje
- při nejistotě ZAOKROUHLI NAHORU (levná chyba > drahá chyba)
"""
import time
import numpy as np
from semantic_router import Route
from semantic_router.encoders import FastEmbedEncoder

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- 3 FIXNÍ STUPNĚ (Bloomova taxonomie), seedované příklady CZ + EN ---
trivialni = Route(name="trivialni", utterances=[
    "kolik je hodin",
    "jaké je dnes datum",
    "co je hlavní město Francie",
    "přelož slovo 'apple' do češtiny",
    "kolik je 15 + 27",
    "what is the capital of Japan",
    "convert 5 km to miles",
    "how do you say 'thank you' in German",
    "ahoj, jak se máš",
    "what day is it today",
])

stredni = Route(name="stredni", utterances=[
    "vysvětli mi rozdíl mezi TCP a UDP",
    "napiš funkci, co seřadí seznam čísel",
    "shrň mi tenhle odstavec do tří vět",
    "jak funguje garbage collection v Javě",
    "proč mi tenhle kód hází null pointer exception",
    "explain the difference between a process and a thread",
    "write a function that reverses a string",
    "summarize the main idea of this article",
    "how do I center a div in CSS",
    "compare REST and GraphQL",
])

tezky = Route(name="tezky", utterances=[
    "přepiš mi celý tenhle modul a navrhni lepší architekturu",
    "navrhni mi škálovatelný systém pro zpracování milionů zpráv za sekundu",
    "rozeber mi tuhle pětistránkovou úvahu a najdi logické chyby",
    "zdůvodni, jestli se vyplatí přejít z monolitu na mikroslužby v našem případě",
    "naplánuj mi kompletní migrační strategii databáze bez výpadku",
    "design a distributed rate limiter that works across multiple regions",
    "refactor this entire codebase to use dependency injection and explain the tradeoffs",
    "analyze this legal contract and identify the risky clauses",
    "prove that this algorithm has O(n log n) complexity",
    "write a detailed technical design doc for a real-time collaboration engine",
])

ROUTES = [trivialni, stredni, tezky]

LEVELS = ["trivialni", "stredni", "tezky"]

def build_index(encoder):
    """Předpočítej embeddingy všech příkladů (jednou)."""
    texts, labels = [], []
    for route in ROUTES:
        for u in route.utterances:
            texts.append(u); labels.append(route.name)
    embs = np.array(encoder(texts), dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    return embs, labels

def classify(encoder, index, q):
    """Skóre ke 3 stupňům (max cosine k příkladům) -> argmax + margin.
    Při nejistotě ZAOKROUHLI NAHORU. Vrací (stupen, info)."""
    embs, labels = index
    v = np.array(encoder([q])[0], dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    sims = embs @ v
    best = {lvl: 0.0 for lvl in LEVELS}
    for lbl, s in zip(labels, sims):
        best[lbl] = max(best[lbl], float(s))
    ranked = sorted(LEVELS, key=lambda l: best[l], reverse=True)
    top, second = ranked[0], ranked[1]
    margin = best[top] - best[second]
    scores_str = " ".join(f"{l[:4]}={best[l]:.2f}" for l in LEVELS)

    # Asymetrie chyb: 'trivialni' jde rovnou na levný model (BEZ pojistky) -> dej ho JEN
    # když je to jasně triviální. 'tezky' pálí Opus -> taky jen když jistý. Cokoli
    # jiného -> 'stredni', kde to přebere Haiku (to JE ta pojistka, escaluje si sám).
    CONF, BIGMARGIN = 0.45, 0.15
    confident = best[top] >= CONF or margin >= BIGMARGIN
    if top == "trivialni" and confident:
        decided = "trivialni"
    elif top == "tezky" and confident:
        decided = "tezky"
    else:
        decided = "stredni"
    tag = "" if decided == top else f"  -> stredni (default/nejisté)"
    return decided, f"[{scores_str}]{tag}"

def main():
    print(f"Loading encoder: {EMBED_MODEL} (první běh stahuje ~470 MB)...")
    t0 = time.time()
    encoder = FastEmbedEncoder(name=EMBED_MODEL)
    encoder.score_threshold = 0.0
    index = build_index(encoder)
    print(f"  ready in {time.time()-t0:.1f}s\n")

    # testovací dotazy — schválně NEJSOU v příkladech, mix CZ/EN, mix těžkosti
    tests = [
        "kolik je hlavní město Peru",              # triviální (CZ, neviděný)
        "what's 8 times 7",                         # triviální (EN)
        "jak zapnu tmavý režim ve Windows",         # triviální/střední
        "napiš mi rekurzivní fibonacci v Pythonu",  # střední (CZ)
        "why is my docker container exiting immediately",  # střední (EN)
        "navrhni mi kompletní architekturu pro chatovací appku s miliony uživatelů",  # těžký (CZ)
        "prove the halting problem is undecidable", # těžký (EN)
        "aha",                                       # nejednoznačné -> nahoru
    ]

    print("dotaz -> stupeň  (čas)   [skóre stupňů]")
    print("-" * 90)
    for q in tests:
        t = time.time()
        stupen, info = classify(encoder, index, q)
        dt = (time.time() - t) * 1000
        print(f"{q[:44]:<46} -> {stupen:<10} ({dt:.0f} ms)  {info}")

if __name__ == "__main__":
    main()
