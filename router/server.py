"""
AI Router — HTTP služba, OpenAI-kompatibilní. Aby router mohl běžet jako systemd
daemon a Open WebUI na něj mohlo ukázat jako na běžný OpenAI endpoint.

  GET  /health                 -> {ok}
  GET  /v1/models              -> jeden model "ai-router" (OWUI si ho vybere)
  POST /v1/chat/completions    -> {messages:[...], stream?} -> odpověď (klasifikuje + routuje)

Klasifikátor se načte JEDNOU při startu. Každý request si přečte čerstvý active_pool.json.
ENV: AI_ROUTER_HOST (default 127.0.0.1), AI_ROUTER_PORT (default 8080).
"""
import os
import json
import time
import uuid
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from difficulty_router import EMBED_MODEL, build_index, FastEmbedEncoder
from availability import Availability
from pool import curated_free
from learning_capture import capture
import router as R

HOST = os.environ.get("AI_ROUTER_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_ROUTER_PORT", "8080"))

print("Načítám klasifikátor…", flush=True)
ENC = FastEmbedEncoder(name=EMBED_MODEL)
ENC.score_threshold = 0.0
INDEX = build_index(ENC)
FREE_POOL = curated_free()
print(f"AI Router služba na http://{HOST}:{PORT}", flush=True)


def route(messages, tools=None, tool_choice=None, stream=False):
    """Vrací (tier, model, result, do_capture).

    `result` je buď text/dict jako dřív, nebo GENERÁTOR kusů textu (stream=True).
    `do_capture(text)` se volá AŽ PO odeslání odpovědi — viz níže.
    """
    t0 = time.time()
    avail = Availability()  # čerstvý active_pool.json každý request
    # R.msg_text, ne m["content"] — u assistant zprávy s tool callem je content None
    # a `len(None)` shazovalo celý request ještě před routováním (2026-08-13).
    hist_chars = sum(len(R.msg_text(m)) for m in messages)
    tier, model, text, info = R.answer(messages, ENC, INDEX, avail, extra_fallback=FREE_POOL,
                                       tools=tools, tool_choice=tool_choice, stream=stream)
    dt = time.time() - t0
    # TRACKING (JP 2026-07-30): viditelné přes `journalctl --user -u router.service -f`
    # Se stream=True je tohle čas do PRVNÍHO kusu, ne celková generace.
    print(f"[timing] {dt:.1f}s{' (ttfb)' if stream else ''}  tier={tier}  model={model!r}  "
          f"history={len(messages)}msg/{hist_chars}chars  tier_info={info}", flush=True)

    user_msgs = [t for m in messages if m.get("role") == "user"
                 and (t := R.msg_text(m))]

    def do_capture(reply_text):
        """TRYCHTÝŘ UČENÍ (R-001): poslední výměna do raw feedu (destiluje Gemini/Hermes).

        MIMO KRITICKOU CESTU (2026-08-12): tohle býval blokující zápis do
        raw/router-conversations.*.md PŘED odesláním odpovědi, takže si za něj uživatel
        počkal. Teď ho volá server až po odeslání. U streamu to ani jinak nejde — text
        v okamžiku rozhodnutí o modelu ještě neexistuje.
        """
        # A tool-call turn has no prose worth distilling — capturing "" would put an empty
        # exchange in the corpus, and the corpus is what the profile is built from.
        #
        # A pomocné úlohy OWUI (`### Task:`) se nezapisují VŮBEC. Šly do korpusu jako
        # `JP: ### Task: Suggest 3-5 relevant follow-up questions…`, tedy šablona OWUI
        # pod štítkem JP — a profil se staví právě z tohohle. Ověřeno 2026-08-16: v
        # zachyceném feedu se střídaly jedna k jedné s jeho skutečnými zprávami.
        if R.is_owui_task(messages):
            return
        if user_msgs and reply_text:
            try:
                capture(user_msgs[-1], reply_text)
            except Exception as e:            # capture nesmí shodit už odeslanou odpověď
                print(f"[capture] selhalo: {str(e)[:120]}", flush=True)

    if hasattr(text, "__next__"):             # generátor -> streamovaná cesta
        return tier, model, text, do_capture
    return tier, model, text if isinstance(text, dict) else (text or ""), do_capture


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p == "/health":
            self._json(200, {"ok": True, "service": "ai-router"})
        elif p == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "ai-router", "object": "model", "owned_by": "jp"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        messages = body.get("messages") or []
        if not messages:
            self._json(400, {"error": "no messages"})
            return
        # TOOLS (2026-08-11): absent -> every byte of the old path is unchanged. Present ->
        # the request is routed only to candidates that can actually call them (router.
        # supports_tools) and the reply carries tool_calls. This is what lets Hermes use the
        # router's model pool and quota guard without silently losing its ability to act.
        tools = body.get("tools") or None
        tool_choice = body.get("tool_choice")
        want_stream = bool(body.get("stream")) and not tools
        # ONE-SHOT PRŮZKUM (2026-08-12): posílá OWUI id konverzace? Router dnes čte jen
        # messages/tools/tool_choice/stream, takže to nikdo neví — a kdyby ho posílalo,
        # odpadne celé hashování při klíčování Claude session. Jeden řádek to rozhodne.
        _ids = {k: v for k, v in self.headers.items()
                if "chat" in k.lower() or "session" in k.lower() or "conversation" in k.lower()}
        _body_ids = {k: body.get(k) for k in ("chat_id", "id", "session_id", "metadata")
                     if k in body}
        if _ids or _body_ids:
            print(f"[owui-id] headers={_ids} body={str(_body_ids)[:200]}", flush=True)
        try:
            tier, model, result, do_capture = route(messages, tools=tools,
                                                    tool_choice=tool_choice, stream=want_stream)
        except Exception as e:
            self._json(500, {"error": f"router: {e}"})
            return

        if hasattr(result, "__next__"):
            self._sse_stream(tier, model, result, do_capture)
            return

        if isinstance(result, dict):
            message = result
            text = result.get("content") or ""
            finish = "tool_calls" if result.get("tool_calls") else "stop"
            # Co agentovi doopravdy odchází. Přidáno 2026-08-14, když Hermes hlásil
            # "Empty response (no content or reasoning)" na odpovědi, které si router
            # logoval jako OK — bez tohohle řádku se ten spor nedal rozhodnout jinak než
            # hádáním. Netiskne obsah, jen tvar.
            _tc = result.get("tool_calls") or []
            _names = [((c.get("function") or {}).get("name")) for c in _tc]
            _offered = sorted((t.get("function") or {}).get("name") for t in (tools or []))
            _unknown = [n for n in _names if n not in _offered]
            print(f"[shape] tools={bool(tools)} finish={finish} "
                  f"content={len(text)}zn tool_calls={len(_tc)}{_names} "
                  f"neznámé={_unknown} nabídnuto={len(_offered)}", flush=True)
        else:
            message = {"role": "assistant", "content": result}
            text = result
            finish = "stop"

        oai = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"ai-router:{tier}:{model}",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        # TADY STÁLO "raději odmítnout než aproximovat" a byla to nejdražší věta v souboru.
        # Tool cally se nestreamovaly a odpověď se poslala jako obyčejné JSON tělo —
        # jenže klient, který požádal o `stream: true`, čte SSE. Naměřeno 2026-08-14
        # Hermesovým vlastním OpenAI klientem proti tomuhle serveru:
        #     stream=True + tools  ->  framů=0  content=0  tool_call_delt=0
        # Agent z toho vidí prázdnou odpověď, třikrát zopakuje týž dotaz a spadne na svůj
        # fallback (tam 429/503) — navenek "Hermes extrémně dlouho přemýšlí". Router si
        # přitom u všech pokusů logoval "pokus OK", takže to nebylo vidět z jeho strany.
        #
        # "Odmítnout" by znamenalo vrátit chybu. Tohle chybu nevracelo, vracelo ticho.
        # Delta framy pro tool call přitom nejsou aproximace: OpenAI formát nevyžaduje,
        # aby se `arguments` posílaly po kouscích — jeden frame s celým řetězcem je platný.
        if body.get("stream"):
            if finish == "tool_calls":
                self._sse_toolcall(oai, message)
            else:
                self._sse(oai, text)
        else:
            self._json(200, oai)
        do_capture(text)

    def _sse_toolcall(self, oai, message):
        """Tool call jako SSE delta framy — tvar, který čekají OpenAI klienti.

        Pořadí je dané: nejdřív frame s rolí (a případným textem, který model napsal
        vedle tool callu), pak frame s tool_calls, pak uzavírací s finish_reason.
        `index` u každého tool callu je povinný — podle něj si klient skládá volání
        dohromady, a bez něj je zahodí stejně tiše, jako se to dělo předtím.
        """
        base = {"id": oai["id"], "object": "chat.completion.chunk",
                "created": oai["created"], "model": oai["model"]}

        def send(delta, finish=None):
            frame = dict(base, choices=[{"index": 0, "delta": delta,
                                         "finish_reason": finish}])
            self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n"
                             .encode("utf-8"))
            self.wfile.flush()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            opener = {"role": "assistant"}
            if message.get("content"):
                opener["content"] = message["content"]
            send(opener)
            calls = []
            for i, c in enumerate(message.get("tool_calls") or []):
                fn = c.get("function") or {}
                calls.append({"index": i, "id": c.get("id"), "type": "function",
                              "function": {"name": fn.get("name"),
                                           "arguments": fn.get("arguments")}})
            if calls:
                send({"tool_calls": calls})
            send({}, finish="tool_calls")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            print("[stream] klient odpojen během tool callu", flush=True)

    def _sse_stream(self, tier, model, chunks, do_capture):
        """SKUTEČNÝ stream — delta framy tak, jak přicházejí od modelu.

        Proti čemu to je (naměřeno 2026-08-12, před touhle změnou):
            bez stream:  TTFB 4,10 s = celkem 4,10 s
            se stream:   TTFB 5,32 s = celkem 5,32 s, 3 framy, celý text v prvním
        `_sse()` níže vyrobil celou odpověď a zabalil ji do JEDNOHO delta framu, takže
        `stream: true` nikdy nic nezrychlilo. Rozdíl proti nativnímu Gemini nebyl v modelu
        ani ve výběru modelu — bylo v tom, že uživatel koukal 4-5 s na prázdno.

        Rotace při 429 zůstává neviditelná: router._call() vytáhne první kus JEŠTĚ uvnitř
        svého try, takže selhávající model vyhodí výjimku dřív, než sem dorazí jediný bajt.
        Odtud dolů už rotovat nelze — proto se to musí stát tam, ne tady.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")   # ať to neschová reverzní proxy
        self.end_headers()
        base = {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion.chunk",
                "created": int(time.time()), "model": f"ai-router:{tier}:{model}"}
        acc = []
        first = True
        try:
            for piece in chunks:
                if not piece:
                    continue
                delta = {"role": "assistant", "content": piece} if first else {"content": piece}
                first = False
                acc.append(piece)
                frame = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}])
                self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()          # bez flushe si to ThreadingHTTPServer nasyslí
            last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
            self.wfile.write(f"data: {json.dumps(last, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Uživatel zavřel záložku uprostřed odpovědi. To, co došlo, se pořád zachytí.
            print("[stream] klient odpojen", flush=True)
        finally:
            do_capture("".join(acc))

    def _sse(self, oai, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        base = {"id": oai["id"], "object": "chat.completion.chunk",
                "created": oai["created"], "model": oai["model"]}
        first = dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": text},
                                     "finish_reason": None}])
        self.wfile.write(f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode("utf-8"))
        last = dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}])
        self.wfile.write(f"data: {json.dumps(last, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
