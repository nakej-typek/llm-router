"""
Claude backend via claude-agent-sdk (2026-07-30) — replaces the `claude -p` subprocess
call for cli:claude:* models. Same subscription auth as `claude -p` (Claude Code as a
library), no extra cost, Article 1 intact.

WHY: subprocess `claude -p` pays a ~3-8s cold-start tax on EVERY message (fresh process,
CLAUDE.md reload) regardless of model size. Benchmarked: ClaudeSDKClient connect() ~0.8s,
then per-turn ~1.7-2.4s floor (some Anthropic-side jitter remains, that's not fixable by
us — same variance a normal Claude Desktop chat has).

ARCHITECTURE: a fresh CLIENT per router call, but since 2026-08-14 that client may
RESUME a per-chat CLI session (`ClaudeAgentOptions.resume`) instead of starting cold.
The old rule — "never reuse a session" — was aimed at a single GLOBAL session, which
would leak context between unrelated OWUI threads; that danger is real and unchanged.
Sessions are therefore keyed per conversation (see "Persistent per-chat sessions"
below) and never shared. On a resumed turn only the NEW messages go into the query;
the CLI already holds the rest, which is the point of fáze 2 — follow-ups stop
re-sending the whole transcript. Persona/architecture/runtime still goes through the
SDK's dedicated `system_prompt` field, not smuggled into the prompt text.

🔒 SAFETY (see A-036 finding — Agent SDK defaults are MORE permissive than `claude -p`,
it will silently RUN Bash with zero prompt on default options):
  - `tools=[]` -> NO built-in tools at all (Bash/Read/Write/Edit/Glob/Grep/WebFetch...).
  - File access comes ONLY via the read-only MCP filesystem server (mcp_fs/fs_server.py,
    whitelisted to ~/syncthing + ~/Projects, no Bash/Write/Edit there either).
  - `allowed_tools` pre-approves ONLY the 4 read-only MCP tools (no prompt needed — the
    whitelist scope IS the safety boundary, nothing here can mutate anything).
Verify after any SDK version bump: tools=[] must still mean zero built-in tools.
"""
import asyncio
import hashlib
import queue
import threading
import os
import time

from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, StreamEvent,
    ResultMessage,
)

ROUTER_VENV_PY = os.path.expanduser("~/.local/share/ai_router/router-venv/bin/python")
FS_SERVER_PY = os.environ.get(
    "ROUTER_MCP_FS_SERVER",
    os.path.expanduser("~/syncthing/archlinux/ai_router/mcp_fs/fs_server.py"))
FS_TOOLS = [f"mcp__jp-filesystem-readonly__{t}"
           for t in ("list_dir", "read_file", "tree", "grep")]

# The "claude_code" system-prompt preset describes the STANDARD built-in toolset (Read,
# Write, Edit, Bash, Glob, Grep) narratively — but tools=[] disables all of them. Without
# this note the model reaches for the built-in "Read"/"Glob"/"Bash" first (as the preset
# primed it to expect) and the call silently fails/produces an empty/truncated turn.
# Tell it explicitly which tools actually exist.
_TOOL_NOTE = (
    "\n\nDŮLEŽITÉ: vestavěné nástroje Read/Write/Edit/Bash/Glob/Grep jsou VYPNUTÉ (nepoužívej "
    "je, selžou). Místo nich máš 4 read-only MCP nástroje pro práci se soubory: "
    "list_dir, read_file, tree, grep (whitelist ~/syncthing + ~/Projects). Použij TYHLE, "
    "když potřebuješ číst/procházet soubory."
)

TIMEOUT_SEC = int(os.environ.get("ROUTER_SDK_TIMEOUT", "120"))


def _flatten_turns(messages):
    """User/assistant history -> one query string (system role handled separately
    via system_prompt, NOT included here)."""
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) == 1:
        return turns[0]["content"]
    lines = []
    for m in turns:
        who = "Uživatel" if m["role"] == "user" else "Asistent"
        lines.append(f"{who}: {m['content']}")
    lines.append("Asistent:")
    return "\n".join(lines)


# ── Persistent per-chat sessions (fáze 2, 2026-08-14) ────────────────────────────────
# WHY: OWUI is stateless towards the backend — it re-sends the ENTIRE transcript on every
# message. Flattened into the query text that meant turn 12 of a chat paid for 11 turns of
# prompt it had already paid for, on the slowest backend we have. `resume` hands that job
# to the CLI's own session store, so a follow-up sends only what is new.
#
# KEYING — why not the OWUI chat id: it may not exist. The one-shot probe in server.py
# (`[owui-id]`) has never printed, though no multi-turn OWUI conversation has gone through
# it since the restart either, so "OWUI sends nothing" is NOT established — measured
# 2026-08-14. The key below needs no cooperation from the client at all: the first user
# message of a conversation never changes while the conversation grows.
#
# COLLISION — two different chats can open with the same message ("ahoj"), and JP opens
# chats that way. A bare first-message hash would splice two conversations into one
# session, which is exactly the context leak the old docstring warned about. So the entry
# also stores a digest per message already sent, and a resume happens ONLY if those are
# still a prefix of what arrived. An edited/regenerated history fails the same check and
# falls back to a cold call — wasteful, never wrong.

_SESS = {}                         # chat key -> _Sess
_SESS_LOCK = threading.Lock()
SESSION_TTL_SEC = int(os.environ.get("ROUTER_SDK_SESSION_TTL", "21600"))   # 6 h
SESSION_MAX = int(os.environ.get("ROUTER_SDK_SESSION_MAX", "200"))


class _Sess:
    __slots__ = ("sid", "digests", "ts")

    def __init__(self, sid, digests, ts):
        self.sid, self.digests, self.ts = sid, digests, ts


def _digest(s):
    return hashlib.sha256((s or "").encode("utf-8", "replace")).hexdigest()[:16]


def _turns(messages):
    return [m for m in messages if m.get("role") in ("user", "assistant")]


def _plan(model, system_prompt, messages):
    """-> (resume_sid_or_None, turns_to_send, key, digests_of_full_history).

    `system_prompt` is deliberately NOT part of the key. router.answer() rebuilds it for
    every single call — `build_system(tier, candidate_label(model), skipped, host)` — so
    it changes whenever the classifier picks a different tier, or a model goes on
    cooldown and the rotation note grows. Keying on it meant a conversation dropped back
    to a cold call for reasons that have nothing to do with which conversation it is;
    measured 2026-08-14, where a chat re-sent 3 and then 7 messages it had already sent.
    Nothing is lost by excluding it: the persona goes to the model through the SDK's
    system_prompt field on every call anyway, and the prefix check below — not the key —
    is what stops two different chats from sharing a session.

    `model` IS part of the key: a session started on haiku must not be resumed on opus.
    """
    turns = _turns(messages)
    digests = [_digest(m.get("content") or "") for m in turns]
    first_user = next((m.get("content") for m in turns if m.get("role") == "user"), "")
    key = _digest(f"{model}\x00{first_user}")
    now = time.time()
    with _SESS_LOCK:
        s = _SESS.get(key)
        if s is None or now - s.ts > SESSION_TTL_SEC:
            return None, turns, key, digests
        sent = len(s.digests)
        # Nothing new (a retry of the same turn) also lands here: resuming and sending an
        # empty query would hang the CLI, so treat it as a cold call.
        if sent >= len(turns) or digests[:sent] != s.digests:
            # Logged, not silent: a prefix that stops matching in normal use would turn
            # every follow-up back into a cold call, and fáze 2 would look implemented
            # while doing nothing. This line is how that shows up.
            print(f"[claude-session] resume přeskočen (prefix {sent}/{len(turns)})",
                  flush=True)
            return None, turns, key, digests
        # A positive line, not just the negative one above: "no skip in the journal" is
        # NOT evidence that a resume happened — a key that never matched logs nothing at
        # all. Measured 2026-08-14, when exactly that ambiguity made a TTFB result
        # unreadable. Both branches now say what they did.
        saved = sum(len(t.get("content") or "") for t in turns[:sent])
        print(f"[claude-session] resume {s.sid[:8]} (+{len(turns) - sent} zpráv, "
              f"ušetřeno {saved} znaků)", flush=True)
        return s.sid, turns[sent:], key, digests


def _remember(key, sid, digests, answer=None):
    """`digests` covers what we SENT; `answer` is what the session then generated.

    The reply must be counted too. The resumed CLI session already holds its own answer,
    so if the stored prefix ended at the user message, the next turn would re-send that
    reply as if the user had typed it. Recorded here because the bug is invisible until
    a two-turn conversation runs against the real CLI.

    The digest is taken over the answer EXACTLY as returned to the client, which is what
    OWUI stores and echoes back next turn. If a client normalises the text the prefix
    check fails and the turn goes cold — wasteful, never wrong, and it prints a line.
    """
    if not sid:
        return
    if answer is not None:
        digests = list(digests) + [_digest(answer)]
    with _SESS_LOCK:
        _SESS[key] = _Sess(sid, digests, time.time())
        if len(_SESS) > SESSION_MAX:
            for k in sorted(_SESS, key=lambda k: _SESS[k].ts)[:len(_SESS) - SESSION_MAX]:
                _SESS.pop(k, None)


def _forget(key):
    with _SESS_LOCK:
        _SESS.pop(key, None)


def _build_opts(model, system_prompt, partial=False, resume=None):
    return ClaudeAgentOptions(
        model=model,
        tools=[],                      # zero built-in tools — see module docstring
        mcp_servers={"jp-filesystem-readonly":
                     {"command": ROUTER_VENV_PY, "args": [FS_SERVER_PY]}},
        allowed_tools=FS_TOOLS,         # pre-approve ONLY these 4 read-only tools
        # PRESET (not a raw string!): a bare string REPLACES the whole built-in Claude
        # Code system prompt, which is what tells the model which tools actually exist —
        # without it, weaker models (Haiku) hallucinate fake tool-call syntax for tools
        # that were never wired up (observed empirically). "append" keeps the real
        # tool-awareness prompt and adds persona/architecture/runtime after it.
        system_prompt={"type": "preset", "preset": "claude_code",
                       "append": (system_prompt or "") + _TOOL_NOTE},
        # Bez tohohle vydá CLI celou odpověď jako JEDEN TextBlock na konci — ověřeno
        # měřením 2026-08-12: streamová cesta dala 2 framy, první až za 20,5s s celým
        # textem. Zapnuté se sypou StreamEvent s raw Anthropic deltami a text teče.
        # Jen pro streamovou cestu; ask() zůstává beze změny.
        include_partial_messages=partial,
        # None -> cold session, exactly as before fáze 2. A string resumes that CLI
        # session and prepends its stored history for free.
        resume=resume,
    )


async def _ask(model, system_prompt, turns, resume=None):
    parts, sid = [], None
    async with asyncio.timeout(TIMEOUT_SEC):
        async with ClaudeSDKClient(
                options=_build_opts(model, system_prompt, resume=resume)) as client:
            await client.query(_flatten_turns(turns))
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            parts.append(block.text)
                elif isinstance(msg, ResultMessage):
                    sid = msg.session_id
    return "".join(parts).strip(), sid


# ── Streaming ────────────────────────────────────────────────────────────────────────
# WHY THIS EXISTS (2026-08-12): server.py streams real delta frames now, but only the
# LiteLLM/Gemini path produced them. Claude fell through to the old fake single-frame
# path — so the SLOWEST backend was the only one that made you wait for the whole answer.
# Measured: a 5-paragraph question routed to cli:claude:opus took 64,0s and arrived in ONE
# frame, while the same question on Gemini arrived in 29 frames spread over 3,5s.
#
# `receive_response()` was already yielding TextBlocks incrementally; _ask() just joined
# them. Nothing new is needed from the SDK — the pieces only had to be let out.
#
# The SDK is async and router.py's _call() is sync, so the async iteration runs in a
# worker thread and hands pieces over a Queue. asyncio.run() cannot be driven from the
# calling thread while it yields.

_DONE = object()


def _start_stream(model, system_prompt, turns, resume):
    """Spawn the worker and pull the first item. -> (first, q, sid_box).

    `first` is a text piece, an exception, or _DONE. Nothing has reached the client yet
    at this point, which is what makes both rotation and the resume fallback possible.
    """
    q = queue.Queue(maxsize=64)
    sid_box = {}

    async def _run():
        try:
            async with asyncio.timeout(TIMEOUT_SEC):
                async with ClaudeSDKClient(options=_build_opts(
                        model, system_prompt, partial=True, resume=resume)) as client:
                    await client.query(_flatten_turns(turns))
                    async for msg in client.receive_response():
                        # S include_partial_messages chodí OBOJE: StreamEvent průběžně a
                        # AssistantMessage s kompletním textem na konci. Brát jen deltas,
                        # jinak by se odpověď poslala dvakrát.
                        if isinstance(msg, StreamEvent):
                            sid_box.setdefault("sid", msg.session_id)
                            ev = msg.event or {}
                            if ev.get("type") == "content_block_delta":
                                d = ev.get("delta") or {}
                                if d.get("type") == "text_delta" and d.get("text"):
                                    q.put(d["text"])
                        elif isinstance(msg, ResultMessage):
                            sid_box["sid"] = msg.session_id
        except BaseException as e:          # včetně TimeoutError/CancelledError
            q.put(e)
        finally:
            q.put(_DONE)

    threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()
    return q.get(), q, sid_box


def ask_stream(model, system_prompt, messages):
    """Generator of text pieces. Raises before yielding anything if the call fails,
    so router.answer() can still rotate to another candidate unseen — same contract as
    the LiteLLM path in router._call()."""
    resume, turns, key, digests = _plan(model, system_prompt, messages)

    # ROTATION SAFETY: první kus vytáhnout ještě tady. Když volání selže, vyhodí se to
    # dřív, než router.answer() cokoli vrátí serveru — takže rotace zůstane neviditelná,
    # stejně jako u LiteLLM cesty.
    first, q, sid_box = _start_stream(model, system_prompt, turns, resume)

    # RESUME FALLBACK: the CLI garbage-collects session files, so a stored id can be dead
    # by the next message. That must never surface as a failed answer — retry cold with
    # the full transcript. Only safe here, before the first byte leaves; a resume that
    # dies mid-stream cannot be retried and is left to fail like any other backend error.
    if resume and isinstance(first, BaseException):
        _forget(key)
        first, q, sid_box = _start_stream(model, system_prompt, _turns(messages), None)

    if isinstance(first, BaseException):
        raise first
    if first is _DONE:
        # EMPTY ANSWER STILL COUNTS AS SENT (2026-08-14). The query reached the CLI, so
        # the session now holds these turns whether or not the model said anything back.
        # Returning early without remembering left the entry frozen at an older prefix,
        # and every later turn re-sent more and more of the transcript — observed in the
        # journal as `ušetřeno` standing still while `+N zpráv` climbed 1 → 3 → 5 → 7.
        _remember(key, sid_box.get("sid"), digests, answer="")
        return iter(())

    def _gen():
        # Accumulated to digest the answer, not to buffer it — every piece is yielded
        # the moment it arrives, which is the whole point of the streaming path.
        acc = [first]
        yield first
        while True:
            item = q.get()
            if item is _DONE:
                # Only reached when the client consumed the whole stream. A disconnect
                # mid-answer leaves the session unremembered, so the next turn is cold —
                # correct, since we would not know what the client actually kept.
                _remember(key, sid_box.get("sid"), digests, answer="".join(acc))
                return
            if isinstance(item, BaseException):
                raise item
            acc.append(item)
            yield item

    return _gen()


def ask(model, system_prompt, messages):
    """Sync entry point for router.py's _call(). Resumes this chat's CLI session when
    one is alive, otherwise starts cold."""
    resume, turns, key, digests = _plan(model, system_prompt, messages)
    try:
        text, sid = asyncio.run(_ask(model, system_prompt, turns, resume))
    except BaseException:
        if not resume:
            raise
        _forget(key)                        # dead session id — see RESUME FALLBACK above
        text, sid = asyncio.run(_ask(model, system_prompt, _turns(messages), None))
    _remember(key, sid, digests, answer=text)
    return text
