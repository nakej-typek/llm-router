"""
End-to-end router V1: dotaz -> klasifikuj -> vyber model -> zavolej -> odpověď.
Gateway = LiteLLM. Provider = Gemini (free, klíč z voice.py .api_key).

Článek 1: jen free flash modely. Voice-safe: NIKDY nevolat gemini-3.1-flash-lite ani
gemini-3-flash-preview (ty patří voice.py) — každý model má vlastní kvótu.

Použití:
  python router.py "tvůj dotaz"
  python router.py            # spustí pár testů
"""
import os
import sys
import json
import re
import time
import shutil
import socket
import pathlib
import subprocess
import urllib.request
import urllib.error
import litellm
from difficulty_router import EMBED_MODEL, ROUTES, build_index, classify, FastEmbedEncoder, LEVELS
from availability import Availability
from pool import curated_free
from persona import build_system
import claude_sdk_backend

litellm.suppress_debug_info = True
os.environ["LITELLM_LOG"] = "ERROR"

# Gemini key: from env GEMINI_API_KEY, else a local .api_key file.
# The key is NEVER committed and never enters the synced tree (see .gitignore / .stignore).
if not os.environ.get("GEMINI_API_KEY"):
    _key_file = pathlib.Path(os.environ.get("GEMINI_KEY_FILE")
                             or pathlib.Path(__file__).parent / ".api_key")
    if _key_file.exists():
        os.environ["GEMINI_API_KEY"] = _key_file.read_text(encoding="utf-8").strip()

# OpenRouter klíč (volitelný) — free registrace. Soubor .openrouter_key vedle scriptu, nebo env.
_or_key = pathlib.Path(__file__).parent / ".openrouter_key"
if _or_key.exists() and not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = _or_key.read_text(encoding="utf-8").strip()

# stupeň -> SEZNAM kandidátů (nejpreferovanější první). Router vezme prvního DOSTUPNÉHO
# (dle guardu active_pool.json, nebo lokálně), při 429 rotuje na dalšího.
# KVÓTOVÁ SEPARACE (Arch A-027): router NESMÍ použít modely sdílené s jinými službami —
# gemini-3.1-flash-lite (voice), gemini-3.5-flash-lite (watcher), flash-latest/-lite-latest
# (Hermes). Čisté na obou klíčích (Win i Arch): gemini-3.5-flash + gemini-3.6-flash.
# ⚠️ 2026-08-10 (A-051/A-053): the list above treats flash-latest/-lite-latest as a bucket of
# their own. They are NOT — aliases bill against a versioned model row (proven: a 429 for
# gemini-flash-latest named "model: gemini-3.6-flash"). WHICH row -lite-latest lands on is
# UNVERIFIED. Also unverified: whether the two keys are even in different Google PROJECTS —
# quotas are per project, not per key, and A-042 only compared key fingerprints. If both keys
# live in one project, this whole separation scheme never existed. Check before relying on it.
# ⚠️ ARCH (A-028/A-032): gemini-3.5-flash/3.6-flash jsou PLACENÉ na JP klíči → 429. Free jsou jen
# ALIASY (flash-latest/-lite-latest). + JP chce celou Claude rodinu (haiku/sonnet/opus, flat sub).
# Env-driven, ať se to přestane přepisovat: ROUTER_GEMINI_SMALL/BIG (aliasy defaultně).
_GS = os.environ.get("ROUTER_GEMINI_SMALL", "gemini/gemini-flash-lite-latest")
_GB = os.environ.get("ROUTER_GEMINI_BIG", "gemini/gemini-flash-latest")
# ⚠️ LATENCY (JP 2026-07-30, měřeno empiricky): Gemini/OpenRouter = skutečné API, ~0.7s.
# JAKÝKOLI cli:claude:* i remote:codex = subprocess/CLI cesta s ~3-8s tax (studený start +
# subscription-auth flow) — Haiku na tom není o nic rychlejší než Opus, daň je ve spuštění
# CLI, ne ve velikosti modelu. Proto: default = rychlý Gemini i pro 'stredni'; Claude/GPT jen
# jako záloha (guard nedostupnost) NEBO explicitním override (viz parse_override níže).
TIER_CANDIDATES = {
    "trivialni": [_GS, _GB, "cli:claude:haiku"],
    # střední: SONNET VEDE od 2026-08-14 — JP rozhodl ("klidně ať to pak používá sonnet
    # a pak opus… sice to bude pomalejší, ale aspoň už to bude funkčnější"), poté co
    # viděl, že Hermes přes router sice routuje, ale `trivialni` i `stredni` vedly TÝMŽ
    # flash-lite, takže se prakticky nic nezměnilo.
    #
    # VĚDOMĚ SE TÍM RUŠÍ MĚŘENÍ Z 2026-08-12, které pořadí otočilo opačně. Nechávám ho
    # tu celé, protože je pořád pravdivé a je to cena, kterou tahle změna platí:
    #   time-to-first-token na stejný dotaz: gemini-flash-lite-latest 0,61s,
    #   gemini-flash-latest 4,17s (alias na 3.6 Flash, přemýšlí před psaním);
    #   kvóty: flash-latest RPD 20 (router na něm dělal 28 volání/den, třetina odmítnuta),
    #   lite RPD 500. Naměřený TTFB Claude cesty: ~5,5s studeně, ~7s při dlouhé historii.
    # Tj. běžný dotaz v OWUI teď čeká ~5s místo ~0,6s. To NENÍ regrese, je to zvolený
    # kompromis: rozlišení kvality nad rychlostí. Vrátit = prohodit _GS na první místo.
    #
    # POZOR na to, co tahle změna NEDĚLÁ: požadavky S NÁSTROJI (tj. Hermesovy agentní
    # tahy) sonneta nikdy nedostanou — supports_tools() vyřazuje cli:*/remote:*, protože
    # ty tool cally tiše zahodí (A-067). Tool turny dál obsluhuje Gemini řada.
    "stredni":   ["cli:claude:sonnet", _GS, _GB, "remote:codex"],
    # těžký: zůstává kvalita napřed — sem se klasifikátor trefí jen při vysoké jistotě (řídké,
    # záměrné "těžké" dotazy), tam se čekání na Opus vyplatí.
    "tezky":     ["cli:claude:opus", "cli:claude:sonnet", "remote:codex", _GB],
}

# Explicitní override (JP 2026-07-30): "opus: ..." / "@opus ..." na začátku zprávy přeskočí
# klasifikátor a jde rovnou na daný backend (zbytek tieru zůstává jako fallback řetěz při
# nedostupnosti). Když víš, že chceš kvalitu a je ti jedno počkat, řekneš si o ni rovnou —
# místo spoléhání na to, že klasifikátor uhodne "těžký".
_OVERRIDE_MODELS = {
    "opus": "cli:claude:opus", "sonnet": "cli:claude:sonnet",
    "haiku": "cli:claude:haiku", "fable": "cli:claude:fable",
    "gpt": "remote:codex", "codex": "remote:codex",
    # "gemini:" míří na LITE od 2026-08-12. Dřív ukazoval na _GB (flash-latest = 3.6 Flash),
    # který před psaním přemýšlí ~4s — což není, co si člověk představí, když si vyžádá
    # "rychlé gemini". Kdo chce ten velký, napíše si o něj tierem nebo "gemini-big:".
    "gemini": _GS, "gemini-big": _GB,
}
_OVERRIDE_RE = re.compile(
    r"^\s*[@]?(" + "|".join(_OVERRIDE_MODELS) + r")\s*[:,]?\s+", re.IGNORECASE)


def msg_text(m):
    """Text jedné zprávy, VŽDY str. Používej místo `m["content"]` úplně všude.

    PROČ (naměřeno 2026-08-13): `content` není vždycky řetězec a dvě z těch podob
    posílá běžný klient, ne exotický:

    * `None` — tak vypadá assistant zpráva, která NESE TOOL CALL. Když ji volající
      pošle zpátky (druhý leg tool callu, přesně to, co dělá Hermes), spadlo
      `len(m.get("content", ""))` na `TypeError: object of type 'NoneType' has no
      len()`. Default v `.get` proti tomu nechrání — klíč existuje, hodnota je None.
      Router vracel 500 a v journalu po tom nezbylo nic, protože to padlo před
      routováním. Zpáteční leg tool callu tím byl rozbitý VŽDY.
    * `list` — multimodal části OpenAI formátu (`[{"type":"text",...}, ...]`).
      Nepadalo, ale `len(list)` počítal položky místo znaků, takže odhad délky
      promptu byl nesmysl a `_flatten` by do promptu vlepil repr Pythonu.
    """
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # multimodal: slep textové části, obrázky zahoď
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and p.get("type") == "text")
    return "" if c is None else str(c)


def parse_override(messages):
    """Vrať (forced_model|None, messages_bez_prefixu). Hledá jen v POSLEDNÍ user zprávě."""
    if not messages or messages[-1].get("role") != "user":
        return None, messages
    text = msg_text(messages[-1])
    m = _OVERRIDE_RE.match(text)
    if not m:
        return None, messages
    forced = _OVERRIDE_MODELS[m.group(1).lower()]
    cleaned = list(messages)
    cleaned[-1] = {**cleaned[-1], "content": text[m.end():]}
    return forced, cleaned


CODEX_ENDPOINT = os.environ.get("CODEX_ENDPOINT", "http://100.77.216.21:8901")


def _flatten(messages):
    """Historie -> jeden prompt pro backendy, co neberou pole zpráv (CLI/remote)."""
    if len(messages) == 1:
        return msg_text(messages[0])
    lines = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            lines.append(msg_text(m))          # profil = preambule bez štítku
        else:
            # "tool" má vlastní štítek: jinak by výsledek nástroje vypadal jako by ho
            # řekl asistent. Sem se dostane jen přes /v1 od klienta s nástroji.
            who = {"user": "Uživatel", "tool": "Nástroj"}.get(role, "Asistent")
            lines.append(f"{who}: {msg_text(m)}")
    lines.append("Asistent:")
    return "\n".join(lines)


def is_owui_task(messages):
    """Je tohle pomocná úloha Open WebUI (název chatu, tagy, návrhy dotazů)?

    OWUI posílá po KAŽDÉ zprávě navíc jednu žádost z vlastní šablony: jediná user zpráva,
    žádný system prompt, obsah začíná `### Task:` a pokračuje `### Guidelines:`. Není to
    nic, co by JP napsal, ani nic, co by měl vidět.

    NAMĚŘENO 2026-08-16, a byl to důvod, proč se JP zdálo, že "Hermes běží na malém
    modelu". Rozpad tahů za šest hodin:
        28x claude:sonnet   history=1msg/~4000 znaků, pokaždé týž otisk klasifikátoru
        19x flash-lite      history=30-50 zpráv/41-85 tisíc znaků   <- skutečná konverzace
    Tedy přesně obráceně, než se čekalo: Sonnet, zapnutý den předtím kvůli KVALITĚ, mlel
    názvy chatů, zatímco agent přemýšlel na flash-lite. A stálo to 5,5 s latence na každou
    zprávu, protože ta pomocná úloha běží dřív, než uživatel uvidí odpověď.

    Druhý důsledek byl v korpusu: `learning_capture` tyhle prompty zapisoval jako
    `JP: ### Task:`, tedy šablonu OWUI pod štítkem JP — stejná třída kontaminace jako
    session routeru vyříznuté 2026-08-14.
    """
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) != 1 or any(m.get("role") == "system" for m in messages):
        return False
    return msg_text(turns[0]).lstrip().startswith("### Task:")


def quota_detail(err):
    """Z 429 vytáhni, KTERÁ kvóta padla — "RPD 20 na gemini-3.7-flash" místo 40 řádků JSONu.

    Proč to existuje: chyba se do journalu logovala oříznutá na 120 znaků, takže tam
    zbylo `litellm.RateLimitError: geminiException - {` a nic víc. Jediná informace, kvůli
    které se na 429 vůbec kouká — jestli došel denní strop, nebo jen minutový — se
    zahazovala. Ověřeno 2026-08-15 na JP-ově výpisu, kde `gemini-flash-latest` ukázal
    `quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20,
    model: gemini-3.7-flash` (alias se tedy mezitím posunul z 3.6 na 3.7 — proto se
    quotaId čte z odpovědi a nikde se nehardcoduje).
    """
    s = str(err)
    qid = re.search(r'"quotaId":\s*"([^"]+)"', s) or re.search(r"quotaId:\s*([\w-]+)", s)
    val = re.search(r'"quotaValue":\s*"?(\d+)', s) or re.search(r"limit:\s*(\d+)", s)
    mod = re.search(r'"model":\s*"([^"]+)"', s) or re.search(r"model:\s*([\w.-]+)", s)
    bits = []
    if qid:
        per = qid.group(1)
        bits.append("denní" if "PerDay" in per else "minutový" if "PerMinute" in per else per)
    if val:
        bits.append(f"limit {val.group(1)}")
    if mod:
        bits.append(f"řádek {mod.group(1)}")
    return " ".join(bits)


class EmptyReply(RuntimeError):
    """Model odpověděl bez textu i bez tool callu.

    Vlastní typ, a ne obyčejná výjimka, kvůli tomu, co s ní dělá answer(): tam se KAŽDÁ
    výjimka bere jako 429 a modelu se nastaví cooldown. Prázdná odpověď ale není rate
    limit — dát za ni flash-lite šedesátivteřinový trest by při agentní smyčce vyřadilo
    zdravý model z poolu po pár tazích. Rotovat ano, trestat ne.
    """


def _delta_text(chunk):
    """Text of one LiteLLM stream chunk, or "" — chunks legitimately carry no content
    (role-only opener, finish_reason-only closer), so this must never raise."""
    try:
        return getattr(chunk.choices[0].delta, "content", None) or ""
    except Exception:
        return ""


def _call(model, messages, tools=None, tool_choice=None, stream=False):
    """Tři zásuvky (dostávají CELOU historii = konverzační paměť):
      - REMOTE 'remote:codex' = HTTP na Archův GPT endpoint (ChatGPT předplatné, 5h okno).
      - INTERFACE 'cli:<tool>[:<variant>]' = lokální subprocess přes PŘEDPLATNÉ (ne API):
          cli:claude:opus  -> claude -p --model opus   (Claude Code předplatné)
      - API (jinak) přes LiteLLM (Gemini / OpenRouter free) — bere pole zpráv nativně.
    Vrací (text, label)."""
    if model == "remote:codex":
        body = json.dumps({"prompt": _flatten(messages)}).encode("utf-8")
        req = urllib.request.Request(
            CODEX_ENDPOINT + "/ask", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=200) as r:
                data = json.load(r)
            return (data.get("answer") or "").strip(), "gpt-codex"
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8", "replace")
            if e.code == 429:
                raise RuntimeError("GPT 5h okno vyčerpáno (429)")  # -> cooldown + rotace
            raise RuntimeError(f"codex endpoint HTTP {e.code}: {payload[:150]}")
    if model.startswith("cli:"):
        parts = model.split(":")           # ["cli","claude","opus"] / ["cli","codex"]
        tool = parts[1]
        variant = parts[2] if len(parts) > 2 else None
        if tool == "claude":
            # SDK backend (2026-07-30, replaces subprocess `claude -p`): a FRESH
            # ClaudeSDKClient per call, which still avoids the ~3-8s CLI cold-start tax
            # that hit EVERY message regardless of model size. (The word "persistent"
            # stood here until 2026-08-12 and was wrong — claude_sdk_backend.ask() calls
            # asyncio.run() with a new client every time. Nothing is persisted; only the
            # ~0.8s connect is saved versus the old subprocess path.) Safety (tools=[],
            # MCP fs whitelist, no Bash/Write/Edit) lives in claude_sdk_backend.py — see
            # its module docstring + A-036/A-037 notes.
            sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
            system_prompt = "\n\n".join(sys_msgs) if sys_msgs else None
            label = f"claude:{variant or 'default'}"
            if stream:
                # Claude is the SLOWEST backend, so it is the one that most needs this.
                # Measured before: a 5-paragraph question took 64,0s and arrived in ONE
                # frame, while Gemini streamed the same answer in 29 frames.
                return claude_sdk_backend.ask_stream(variant or "sonnet",
                                                     system_prompt, messages), label
            return claude_sdk_backend.ask(variant or "sonnet", system_prompt, messages), label
        exe = shutil.which(tool) or tool
        prompt = _flatten(messages)
        if tool == "codex":
            cmd = [exe, "exec", "--sandbox", "read-only"] + (["--model", variant] if variant else []) + [prompt]
        else:
            raise RuntimeError(f"neznámý CLI nástroj: {tool}")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=300)
        if res.returncode != 0:
            raise RuntimeError(f"{tool} CLI rc={res.returncode}: {(res.stderr or '')[:200]}")
        return res.stdout.strip(), f"{tool}:{variant or 'default'}"
    # TOOLS (2026-08-11): LiteLLM does the whole OpenAI<->Gemini function-calling
    # translation itself — `functionDeclarations` out, `functionCall` back, and
    # role:"tool" -> `functionResponse` on the return leg. Verified live on JP's key
    # (A-069): both legs, unmodified, on a free model. There is nothing for us to write.
    #
    # The router could not serve a tool-using agent for one reason only: the line below
    # used to end at `.message.content`, throwing away `tool_calls` one attribute short.
    kw = {"model": model, "messages": messages, "timeout": 60}
    if tools:
        kw["tools"] = tools
        if tool_choice is not None:
            kw["tool_choice"] = tool_choice

    # STREAMING (2026-08-12). Measured before this existed: TTFB 4,10s == total 4,10s, and
    # with stream:true 5,32s == 5,32s in 3 SSE frames with the whole answer in the first —
    # server.py FAKED the stream. Router overhead was ~0 (its own [timing] said 4,3s of a
    # 4,3s request), so the gap against native Gemini/Claude was never the model choice or
    # the classifier: it was staring at nothing until generation finished.
    #
    # ROTATION SAFETY — the reason the first chunk is pulled HERE and not in server.py:
    # answer() rotates to the next candidate when a call raises (429, overload). If we
    # returned the raw iterator, the 429 would surface only after server.py had already
    # written frames to the client, and the rotation would appear mid-answer as garbage.
    # Pulling one chunk inside this try means a failing model still raises before a single
    # byte reaches the user, so rotation stays invisible exactly as it is today.
    if stream and not tools:
        it = litellm.completion(**kw, stream=True)
        first = next(it, None)          # raises here on 429 -> answer() rotates, nothing sent

        def _chunks():
            if first is not None:
                yield _delta_text(first)
            for ch in it:
                yield _delta_text(ch)

        return _chunks(), model.split("/", 1)[-1]

    r = litellm.completion(**kw)
    msg = r.choices[0].message
    if not tools:
        if not (msg.content or "").strip():
            raise EmptyReply(f"{model} vrátil prázdný text "
                             f"(finish_reason={getattr(r.choices[0], 'finish_reason', '?')})")
        return msg.content, model.split("/", 1)[-1]
    # With tools in play the caller needs the whole message, not just prose.
    calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        calls.append({
            "id": tc.id,          # ⚠ VERBATIM. Gemini embeds a thought signature here:
                                  # ~90 chars including '/' and '+' (A-069). Truncating,
                                  # regex-cleaning or regenerating it breaks the SECOND leg
                                  # while leg 1 still looks perfect — a silent failure.
                                  # api_server.py:1759 scrubs a header id; do NOT copy that
                                  # pattern here.
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        })
    out = {"role": "assistant", "content": msg.content}
    if calls:
        out["tool_calls"] = calls
    # PRÁZDNÁ ODPOVĚĎ NENÍ ÚSPĚCH (2026-08-14). Bez tohohle se `{"content": None}` bez
    # jediného tool callu vrátilo volajícímu jako platná odpověď. Naměřeno na Hermesovi
    # hned první den, co jede přes router: v jeho journalu tři řádky
    # "Empty response (no content or reasoning) — retry 1..3/3" a pak fallback, zatímco
    # router si u všech tří pokusů spokojeně logoval "pokus OK". Pro JP to vypadalo jako
    # že Hermes extrémně dlouho přemýšlí; ve skutečnosti třikrát dostal nic.
    # Rotace na dalšího kandidáta je správná reakce — jiný model odpoví.
    if not (msg.content or "").strip() and not calls:
        raise EmptyReply(f"{model} vrátil prázdnou odpověď "
                         f"(finish_reason={getattr(r.choices[0], 'finish_reason', '?')})")
    return out, model.split("/", 1)[-1]


FOLLOWUP_MAX_WORDS = 6  # kratší uživatelská zpráva = pravděpodobně follow-up


def decide_tier(messages, encoder, index):
    """Stupeň s HYSTEREZÍ: krátký follow-up neklesne pod téma vlákna.
    Klasifikuje poslední dotaz; když je krátký, zdědí (max) stupeň poslední PODSTATNÉ
    uživatelské zprávy — aby náročná diskuze zůstala na silném modelu i přes krátké
    doplňující otázky ("a proč?"). Nahoru eskaluje vždy volně."""
    user_msgs = [msg_text(m) for m in messages if m.get("role") == "user"]
    last = user_msgs[-1]
    tier, info = classify(encoder, index, last)
    if len(user_msgs) >= 2 and len(last.split()) < FOLLOWUP_MAX_WORDS:
        for prev in reversed(user_msgs[:-1]):
            if len(prev.split()) >= FOLLOWUP_MAX_WORDS:  # poslední podstatná = téma vlákna
                thread_tier, _ = classify(encoder, index, prev)
                if LEVELS.index(thread_tier) > LEVELS.index(tier):
                    info += f"  [hystereze: follow-up {tier}->{thread_tier}]"
                    tier = thread_tier
                break
    return tier, info


def candidate_label(model):
    """Přátelský název modelu pro runtime blok."""
    if model == "remote:codex":
        return "GPT / Codex (ChatGPT předplatné, přes Arch)"
    if model == "cli:claude:opus":
        return "Claude Opus (tvoje předplatné, přes claude -p)"
    if model.startswith("gemini/"):
        return model.split("/")[-1] + " (free Gemini)"
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1] + " (OpenRouter free)"
    return model


def supports_tools(model):
    """Can this candidate participate in an OpenAI tool-call exchange?

    Two filters, deliberately different in kind:

    1. STRUCTURAL — `cli:*` and `remote:*` are subprocess/HTTP paths that take a flattened
       prompt and hand back text. They cannot carry `tools` in either direction. Sending a
       tool request to one does NOT error: it answers fluently and drops the tools on the
       floor, so an agent on the other end simply stops being able to act while still
       sounding fine (A-067). That silence is why this filter is structural rather than a
       best-effort check.
    2. DATA-DRIVEN — for API models, ask LiteLLM's own registry. Evaluated per candidate at
       routing time, never cached: the registry ships with the library and updates with it,
       whereas our hand-maintained model lists have drifted from reality twice (A-027's
       quota-bucket comments, A-028's "versioned = paid").
    """
    if model.startswith("cli:") or model.startswith("remote:"):
        return False
    try:
        return bool(litellm.supports_function_calling(model=model))
    except Exception:
        return False  # unknown to the registry -> do not risk a silent drop


def answer(messages, encoder, index, avail, extra_fallback=(), tools=None, tool_choice=None,
           stream=False):
    """messages = [{"role":"user"/"assistant","content":...}]. Stupeň dle decide_tier
    (s hysterezí follow-upů); historii předá vybranému modelu. Explicitní model-prefix
    ("opus: ...") přeskočí klasifikátor a jde rovnou na daný backend."""
    forced, messages = parse_override(messages)
    if is_owui_task(messages):
        # Pomocná úloha OWUI: klasifikátor se přeskakuje úplně (ušetří i embedding) a jde
        # se na nejlevnější rychlý model. Vygenerovat název chatu nepotřebuje Sonnet ani
        # úsudek o obtížnosti — viz is_owui_task(). Tier zůstává "trivialni", takže si
        # zachová celý svůj rotační řetěz, kdyby flash-lite nebyl k dispozici.
        tier, info = "trivialni", "[OWUI pomocná úloha — klasifikátor přeskočen]"
    else:
        tier, info = decide_tier(messages, encoder, index)
    host = os.environ.get("AI_ROUTER_HOST_LABEL") or socket.gethostname()
    if forced:
        info += "  [explicitní override]"
        candidates = [forced] + [m for m in TIER_CANDIDATES[tier] if m != forced] + list(extra_fallback)
    else:
        candidates = list(TIER_CANDIDATES[tier]) + list(extra_fallback)
    if tools:
        # A tool-carrying request must never reach a candidate that would drop the tools and
        # answer anyway — see supports_tools().
        #
        # THE TIER MAP IS ADVISORY HERE, NOT BINDING. Measured on the live map (A-070): after
        # filtering, "tezky" has exactly ONE capable candidate and it is the RPD-20 row that
        # is spent most days. Honouring the tier strictly would mean a hard tool request has
        # a single option on the most exhausted model in the system, and the "no capable
        # candidate" error would be the normal outcome rather than the exceptional one.
        #
        # So: keep the tier's own capable candidates first (its judgement about difficulty is
        # still worth something), then fall through to every other capable model we know of.
        # Tool support is a hard constraint; tier is a preference. Only a request with no
        # capable model anywhere is a real failure.
        seen = set(candidates)
        elsewhere = []
        for _tier_models in TIER_CANDIDATES.values():
            for _m in _tier_models:
                if _m not in seen:
                    seen.add(_m)
                    elsewhere.append(_m)
        incapable = [m for m in candidates if not supports_tools(m)]
        candidates = ([m for m in candidates if supports_tools(m)]
                      + [m for m in elsewhere if supports_tools(m)])
        if not candidates:
            raise RuntimeError(
                "request carries tools but no known model can call them "
                f"(excluded: {', '.join(m.split('/')[-1] for m in incapable)})")
        if incapable:
            skipped_note = ", ".join(m.split("/")[-1] for m in incapable)
            print(f"[tools] vyřazeno (neumí nástroje): {skipped_note}", flush=True)
    skipped = []
    last_err = None
    for model in candidates:
        ok, why = avail.available(model)
        if not ok:
            skipped.append(f"{model.split('/')[-1]}({why})")
            continue
        # PERSONA + RUNTIME pro VÍTĚZE (model, který se právě volá) — profil + architektura + runtime
        #
        # ALE NE, KDYŽ SI VOLAJÍCÍ PŘINESL SVŮJ VLASTNÍ SYSTEM PROMPT (2026-08-14).
        # Router vznikl pro OWUI, kde system zpráva nechodí a persona je jediná identita,
        # kterou model dostane. Od chvíle, kdy je router poskytovatelem pro Hermes agenta,
        # to přestalo platit: Hermes posílá svůj vlastní system prompt s popisem nástrojů
        # a skillů. Vlepit před něj "jsi ai-router JP" znamená dát modelu dvě neslučitelné
        # identity a rozředit instrukce, na kterých stojí tool calling — a tool calling JE
        # ten mechanismus, kterým se Hermes učí. Kdo si přinese system prompt, ten ví, čím
        # chce být; my mu do toho nemluvíme.
        has_own_system = any(m.get("role") == "system" for m in messages)
        if has_own_system:
            # ...ale JEDNU faktickou větu mu přidat musíme: který model tenhle tah
            # doopravdy obsluhuje. Router je jediné místo v systému, kde je to známo —
            # agent zná jen jméno, které má v konfiguraci ("ai-router"), a to JP nic
            # neříká ("jako informace ai-router je mi upřímně docela k ničemu", 2026-08-14).
            # Připojuje se NA KONEC jeho vlastního system promptu, ne před něj: identita
            # zůstává jeho, tohle je jen runtime fakt, který jinak nemá odkud vzít.
            call_messages = []
            _stamped = False
            for m in messages:
                if m.get("role") == "system" and not _stamped:
                    m = {**m, "content": f"{msg_text(m)}\n\n"
                         f"[ROUTER] Tenhle tah obsluhuje model: {candidate_label(model)}. "
                         f"Když máš uvést, na čem běžíš, uveď TOHLE jméno — ne jméno, "
                         f"které máš v konfiguraci."}
                    _stamped = True
                call_messages.append(m)
        else:
            sysmsg = build_system(tier, candidate_label(model), skipped, host)
            call_messages = [sysmsg] + messages
        prompt_chars = sum(len(msg_text(m)) for m in call_messages)
        t_call = time.time()
        try:
            text, used = _call(model, call_messages, tools=tools, tool_choice=tool_choice,
                               stream=stream)
            avail.record_use(model)
            note = f"   [rotace, přeskočeno: {', '.join(skipped)}]" if skipped else ""
            # When streaming this is TIME TO FIRST CHUNK, not total generation —
            # _call() returns as soon as the first chunk lands. That is deliberate:
            # TTFB is the number that decides whether this feels like Gemini.
            print(f"[timing]   pokus OK: {model} za {time.time()-t_call:.1f}s "
                  f"(prompt {prompt_chars} chars{', ttfb' if stream else ''})", flush=True)
            return tier, used + note, text, info
        except EmptyReply as e:
            # Rotace BEZ cooldownu — viz EmptyReply. Model je v pořádku, jen tenhle
            # jeden tah nic nevrátil; příští dotaz na něj může klidně jít.
            print(f"[timing]   PRÁZDNÉ: {model} po {time.time()-t_call:.1f}s: "
                  f"{str(e)[:160]}", flush=True)
            skipped.append(f"{model.split('/')[-1]}(prázdná odpověď)")
            last_err = e
            continue
        except Exception as e:
            # Detail kvóty PŘED oříznutím — jinak z 429 zbyde v journalu jen
            # "geminiException - {" a nedá se rozlišit vyčerpaný den od vyčerpané minuty.
            det = quota_detail(e)
            print(f"[timing]   pokus SELHAL: {model} po {time.time()-t_call:.1f}s: "
                  f"{(det + ' | ') if det else ''}{str(e)[:120]}", flush=True)
            # GPT 429 = 5h okno vyčerpáno → dlouhý cooldown; API 429 → krátký (per-minute)
            default_wait = 1800 if model.startswith("remote:") else 60
            wait = avail.record_429(model, e, default_wait=default_wait)
            skipped.append(f"{model.split('/')[-1]}(429→{int(wait)}s)")
            last_err = e
    # ČITELNÁ HLÁŠKA, NE VÝPIS CHYBY (2026-08-15). Tady se dřív vracel `last={last_err}`,
    # což je u 429 od Gemini čtyřicetiřádkový JSON — a ten skončil JP-ovi v chatu jako
    # "odpověď" Hermese. Chybová hláška je taky produkt: musí říct co se stalo, kdy to
    # zkusit znovu, a proč nepomohl Claude.
    wait = ""
    m = re.search(r"(\d+)s\)", " ".join(skipped))       # "…(429→54s)"
    if m:
        wait = f" Zkus to znovu za ~{m.group(1)} s."
    det = quota_detail(last_err) if last_err else ""
    why_no_claude = ""
    if tools and incapable:
        # Tohle je ta část, kterou by JP jinak nepochopil: požádal o Claude, Claude běží,
        # a stejně se nepoužil. Důvod není dostupnost, ale to, že tool cally neunese.
        why_no_claude = (" Claude/GPT se do výběru nedostaly, protože tenhle dotaz nese "
                         "nástroje a ty je neumí přenést (tiše by je zahodily).")
    tried = ", ".join(s.split("(")[0] for s in skipped) or "žádný kandidát"
    msg = (f"[router] Teď nemám volný model. Zkoušel jsem: {tried}."
           f"{(' Kvóta: ' + det + '.') if det else ''}{wait}{why_no_claude}")
    print(f"[timing] ŽÁDNÝ MODEL  tier={tier}  {msg}", flush=True)
    return tier, None, msg, info


def _load():
    print("Načítám klasifikátor…")
    enc = FastEmbedEncoder(name=EMBED_MODEL)
    enc.score_threshold = 0.0
    index = build_index(enc)
    avail = Availability()
    free_pool = curated_free()  # kurátor z watcherova free poolu = záložní kandidáti
    return enc, index, avail, free_pool


def chat():
    """Interaktivní konverzace s PAMĚTÍ historie (router střídá modely, ale kontext drží)."""
    enc, index, avail, free_pool = _load()
    print("\nChat režim — historie se pamatuje. 'exit' ukončí.\n")
    history = []
    while True:
        try:
            q = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break
        history.append({"role": "user", "content": q})
        t = time.time()
        tier, model, text, info = answer(history, enc, index, avail, extra_fallback=free_pool)
        dt = time.time() - t
        print(f"\n[{pretty_model(model)} · {tier} · {dt:.0f}s]")
        print((text or "(prázdné)").strip() + "\n")
        history.append({"role": "assistant", "content": text or ""})


def main():
    if "--chat" in sys.argv:
        chat()
        return
    enc, index, avail, free_pool = _load()
    if len(sys.argv) > 1:
        queries = [" ".join(a for a in sys.argv[1:] if not a.startswith("--"))]
    else:
        queries = ["jaké je hlavní město Francie", "napiš mi v Pythonu funkci, co obrátí řetězec"]

    for q in queries:
        print("\n" + "=" * 74)
        print(f"DOTAZ: {q}")
        t = time.time()
        tier, model, text, info = answer([{"role": "user", "content": q}], enc, index, avail,
                                         extra_fallback=free_pool)
        dt = time.time() - t
        print("-" * 74)
        print(f"  🧠 ODPOVĚDĚL:  {pretty_model(model)}")
        print(f"  📊 stupeň:     {tier}   ({dt:.0f}s)   skóre {info}")
        print("=" * 74)
        print(text.strip() if text else "(prázdné)")


def pretty_model(model):
    if model is None:
        return "(žádný — vše selhalo)"
    base = model.split(" ")[0]
    names = {
        "claude:opus": "Opus  (Claude Code — tvoje předplatné, flat)",
        "gpt-codex": "GPT / Codex  (ChatGPT předplatné přes Arch, 5h okno)",
        "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite  (free)",
        "gemini-3.5-flash": "Gemini 3.5 Flash  (free)",
        "gemini-3.6-flash": "Gemini 3.6 Flash  (free)",
        "gemini-flash-latest": "Gemini Flash Latest  (free)",
    }
    if base in names:
        label = names[base]
    elif ":free" in base:
        label = f"{base}  (OpenRouter free pool)"
    else:
        label = base
    return label + (model[len(base):] if " " in model else "")


if __name__ == "__main__":
    main()
