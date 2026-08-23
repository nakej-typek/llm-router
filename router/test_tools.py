"""test_tools.py — plumbing tests for the tool-calling path (2026-08-11, WIN-CLAUDE).

Runs on Windows with NO Gemini key: `litellm.completion` is stubbed. That is deliberate —
the OpenAI<->Gemini translation is LiteLLM's job and ARCH verified it live on JP's key
(A-069, both legs). What is ours, and therefore what is tested here, is the plumbing:

  - does a tools request reach only candidates that can actually call them
  - does a request WITHOUT tools behave byte-for-byte as before
  - does the tool_call id survive us untouched
  - does the round trip survive server.py's envelope

Run:  python router/test_tools.py
"""
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# router.py imports claude_sdk_backend at module load, which imports claude_agent_sdk at
# module load — an Arch-only dependency. So router.py cannot be imported at all on a machine
# without the Claude SDK, which makes every part of it untestable here, including parts that
# have nothing to do with Claude. Stubbed rather than worked around; worth making that import
# lazy one day, but that is a change to a shared file and not this test's business.
_stub = types.ModuleType("claude_agent_sdk")
# StreamEvent added 2026-08-12 with the streaming work. The stub must track what
# claude_sdk_backend.py actually imports, or this whole file dies at import and the
# Shape 1 tests stop running — silently, since a red import looks like a broken test
# rather than a missing stub name.
for _name in ("ClaudeSDKClient", "ClaudeAgentOptions", "AssistantMessage", "TextBlock",
              "StreamEvent", "ResultMessage"):
    setattr(_stub, _name, type(_name, (), {}))
sys.modules.setdefault("claude_agent_sdk", _stub)

import litellm  # noqa: E402
import router as R  # noqa: E402

# The real id Gemini returned in A-069. Not a short opaque token: it embeds a thought
# signature, runs ~90 chars, and contains '/' and '+'. If anything in our code trims,
# scrubs or regenerates it, leg 2 fails while leg 1 still looks perfect.
REAL_ID = ("EXOvfanf__thought__EjQKMgERTTIPkpMNTttiYfX00r1bt3ZYxr1vBQANLvY4wP7c6jiP0s"
           "avdpUyA0/Ld1eOYr7o")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_disk_free",
        "description": "Free space for a path",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]


def _msg(content=None, tool_calls=None):
    """Shape LiteLLM hands back: attribute access, not dict access."""
    fn = None
    calls = None
    if tool_calls:
        calls = []
        for tid, name, args in tool_calls:
            fn = types.SimpleNamespace(name=name, arguments=args)
            calls.append(types.SimpleNamespace(id=tid, function=fn, type="function"))
    message = types.SimpleNamespace(content=content, tool_calls=calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class SupportsTools(unittest.TestCase):
    def test_cli_and_remote_excluded_structurally(self):
        # These take a flattened prompt and return text. They cannot carry tools either
        # way, and crucially they do not ERROR when asked to — they answer and drop them.
        for m in ("cli:claude:opus", "cli:claude:sonnet", "cli:codex", "remote:codex"):
            self.assertFalse(R.supports_tools(m), m)

    def test_gemini_models_accepted(self):
        for m in ("gemini/gemini-flash-latest", "gemini/gemini-flash-lite-latest",
                  "gemini/gemini-3.5-flash"):
            self.assertTrue(R.supports_tools(m), m)

    def test_unknown_model_is_refused_not_risked(self):
        self.assertFalse(R.supports_tools("gemini/definitely-not-a-real-model-xyz"))


class CallReturnShape(unittest.TestCase):
    def setUp(self):
        self._real = litellm.completion
        self.seen = {}

    def tearDown(self):
        litellm.completion = self._real

    def test_without_tools_returns_plain_text_unchanged(self):
        litellm.completion = lambda **kw: (self.seen.update(kw), _msg(content="ahoj"))[1]
        out, label = R._call("gemini/gemini-flash-latest", [{"role": "user", "content": "x"}])
        self.assertEqual(out, "ahoj")                      # a string, exactly as before
        self.assertEqual(label, "gemini-flash-latest")
        self.assertNotIn("tools", self.seen)               # nothing new sent downstream

    def test_with_tools_returns_message_and_preserves_id_verbatim(self):
        litellm.completion = lambda **kw: (self.seen.update(kw), _msg(
            content=None,
            tool_calls=[(REAL_ID, "get_disk_free", '{"path": "/home"}')]))[1]
        out, _ = R._call("gemini/gemini-flash-latest",
                         [{"role": "user", "content": "free space?"}], tools=TOOLS)
        self.assertIsInstance(out, dict)
        self.assertEqual(self.seen["tools"], TOOLS)
        self.assertEqual(out["tool_calls"][0]["id"], REAL_ID)
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "get_disk_free")
        self.assertEqual(out["tool_calls"][0]["function"]["arguments"], '{"path": "/home"}')

    def test_tool_choice_only_forwarded_when_given(self):
        litellm.completion = lambda **kw: (self.seen.update(kw), _msg(content="x"))[1]
        R._call("gemini/gemini-flash-latest", [{"role": "user", "content": "x"}],
                tools=TOOLS, tool_choice=None)
        self.assertNotIn("tool_choice", self.seen)

    def test_second_leg_text_answer_after_tool_result(self):
        # role:"tool" in the history is LiteLLM's to translate; ours is to not mangle it.
        litellm.completion = lambda **kw: (self.seen.update(kw), _msg(
            content="There is 412 GB of free space on /home."))[1]
        out, _ = R._call("gemini/gemini-flash-latest", [
            {"role": "user", "content": "free space?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": REAL_ID, "type": "function",
                 "function": {"name": "get_disk_free", "arguments": '{"path": "/home"}'}}]},
            {"role": "tool", "tool_call_id": REAL_ID, "content": "412 GB"},
        ], tools=TOOLS)
        self.assertIsNone(out.get("tool_calls"))
        self.assertIn("412 GB", out["content"])
        self.assertEqual(self.seen["messages"][2]["tool_call_id"], REAL_ID)


class EmptyReplyTests(unittest.TestCase):
    """Prázdná odpověď se nesmí vydávat za úspěch.

    Odkud to je: první den, co Hermes jel přes router, mu router třikrát po sobě vrátil
    `{"content": None}` bez tool callu a u všech tří si logoval "pokus OK". Hermes to
    hlásil jako "Empty response — retry 1..3/3" a šel do fallbacku; navenek to vypadalo,
    že extrémně dlouho přemýšlí. Test 3 je ten, na kterém záleží nejvíc: tah, kde model
    vrátí JEN tool call a žádný text, je naprosto normální a musí projít.
    """

    def setUp(self):
        self._real = litellm.completion

    def tearDown(self):
        litellm.completion = self._real

    def test_empty_content_without_tools_raises(self):
        litellm.completion = lambda **kw: _msg(content=None)
        with self.assertRaises(R.EmptyReply):
            R._call("gemini/x", [{"role": "user", "content": "ahoj"}])

    def test_whitespace_only_content_raises(self):
        litellm.completion = lambda **kw: _msg(content="   \n ")
        with self.assertRaises(R.EmptyReply):
            R._call("gemini/x", [{"role": "user", "content": "ahoj"}])

    def test_tool_call_without_text_is_NOT_empty(self):
        """Nejběžnější agentní tah vůbec. Kdyby ho tahle kontrola zamítla, přestal by
        fungovat celý tool calling — a to je horší než chyba, kterou opravuje."""
        litellm.completion = lambda **kw: _msg(
            content=None, tool_calls=[("id1", "skill_manage", '{"action":"create"}')])
        out, _used = R._call("gemini/x", [{"role": "user", "content": "ahoj"}],
                             tools=[{"type": "function"}])
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "skill_manage")
        self.assertIsNone(out["content"])

    def test_empty_with_tools_requested_but_none_returned_raises(self):
        litellm.completion = lambda **kw: _msg(content="")
        with self.assertRaises(R.EmptyReply):
            R._call("gemini/x", [{"role": "user", "content": "ahoj"}],
                    tools=[{"type": "function"}])

    def test_normal_text_still_passes(self):
        litellm.completion = lambda **kw: _msg(content="odpověď")
        text, _used = R._call("gemini/x", [{"role": "user", "content": "ahoj"}])
        self.assertEqual(text, "odpověď")


class EmptyReplyRotationTests(unittest.TestCase):
    """answer() musí na prázdno rotovat, ale NESMÍ modelu dát cooldown.

    V answer() se každá výjimka převádí na record_429(). Prázdná odpověď není rate limit;
    trestat za ni zdravý model by ho v agentní smyčce po pár tazích vyřadilo z poolu.
    """

    class FakeAvail:
        def __init__(self):
            self.cooled = []
        def available(self, model):
            return True, ""
        def record_use(self, model):
            pass
        def record_429(self, model, err=None, default_wait=60):
            self.cooled.append(model)
            return default_wait

    def setUp(self):
        self._real_call = R._call
        self._real_tiers = R.TIER_CANDIDATES
        R.TIER_CANDIDATES = {k: ["gemini/prvni", "gemini/druhy"]
                             for k in self._real_tiers}

    def tearDown(self):
        R._call = self._real_call
        R.TIER_CANDIDATES = self._real_tiers

    def test_rotates_to_next_candidate_without_cooldown(self):
        seen = []

        def fake_call(model, messages, tools=None, tool_choice=None, stream=False):
            seen.append(model)
            if model == "gemini/prvni":
                raise R.EmptyReply("prázdno")
            return "z druhého", model

        R._call = fake_call
        avail = self.FakeAvail()
        enc = index = None
        R.decide_tier = lambda m, e, i: ("stredni", "[test]")
        tier, used, text, info = R.answer(
            [{"role": "user", "content": "ahoj"}], enc, index, avail)
        self.assertEqual(seen, ["gemini/prvni", "gemini/druhy"])
        self.assertEqual(text, "z druhého")
        self.assertEqual(avail.cooled, [], "prázdná odpověď nesmí dát cooldown")


class OwuiTaskTests(unittest.TestCase):
    """Pomocné úlohy OWUI se musí poznat — jinak si berou Sonnet a lezou do korpusu.

    Naměřeno 2026-08-16: 28 tahů na claude:sonnet, všechny `1msg/~4000 znaků` s týmž
    otiskem klasifikátoru = generování názvů chatu. Skutečná konverzace mezitím běžela
    na flash-lite. A `learning_capture` ty šablony ukládal jako `JP: ### Task:`.
    """

    REAL = ("### Task:\nSuggest 3-5 relevant follow-up questions or prompts that the user "
            "might naturally ask next in this conversation as a **user**, based on the chat "
            "history.\n### Guidelines:\n- Write all follow-up questions from the user's "
            "point of view.")

    def test_detects_the_real_owui_template(self):
        self.assertTrue(R.is_owui_task([{"role": "user", "content": self.REAL}]))

    def test_leading_whitespace_still_detected(self):
        self.assertTrue(R.is_owui_task([{"role": "user", "content": "\n  " + self.REAL}]))

    def test_real_conversation_is_not_a_task(self):
        self.assertFalse(R.is_owui_task([{"role": "user", "content": "ahoj, co je router"}]))

    def test_multi_turn_is_not_a_task(self):
        """Skutečná konverzace může začínat čímkoli; rozhoduje i to, že úloha je JEDNOTAH."""
        self.assertFalse(R.is_owui_task([
            {"role": "user", "content": self.REAL},
            {"role": "assistant", "content": "…"},
            {"role": "user", "content": "dál"}]))

    def test_agent_request_with_system_prompt_is_not_a_task(self):
        """Hermes posílá vlastní system prompt. Kdyby jeho tah spadl do téhle větve,
        přišel by o klasifikátor i o silné modely."""
        self.assertFalse(R.is_owui_task([
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": self.REAL}]))

    def test_markdown_heading_that_merely_looks_similar(self):
        self.assertFalse(R.is_owui_task([{"role": "user", "content": "### Taskbar nejde skrýt"}]))

    def test_none_content_does_not_raise(self):
        self.assertFalse(R.is_owui_task([{"role": "user", "content": None}]))


class QuotaDetailTests(unittest.TestCase):
    """Z 429 se musí dát přečíst, KTERÁ kvóta padla.

    Skutečné tělo od Gemini, zkrácené. Do journalu se chyba loguje oříznutá na 120 znaků,
    takže bez tohohle z ní zbyde `geminiException - {` — a rozdíl mezi vyčerpaným dnem
    a vyčerpanou minutou je přesně to jediné, kvůli čemu se na 429 kouká.
    """

    REAL = ('litellm.RateLimitError: geminiException - {"error": {"code": 429, "message": '
            '"You exceeded your current quota... limit: 20, model: gemini-3.7-flash", '
            '"details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", '
            '"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",'
            ' "quotaValue": "20", "quotaDimensions": {"model": "gemini-3.7-flash"}}]}]}}')

    def test_reads_daily_quota(self):
        d = R.quota_detail(self.REAL)
        self.assertIn("denní", d)
        self.assertIn("20", d)
        self.assertIn("gemini-3.7-flash", d)

    def test_reads_per_minute_quota(self):
        d = R.quota_detail('"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"')
        self.assertIn("minutový", d)

    def test_unknown_error_yields_empty_string(self):
        self.assertEqual(R.quota_detail("nějaká úplně jiná chyba"), "")

    def test_never_raises_on_odd_input(self):
        for bad in (None, 0, object(), b"x"):
            R.quota_detail(bad)


class NoModelMessageTests(unittest.TestCase):
    """Hláška "nemám model" končí JP-ovi v chatu jako odpověď Hermese.

    Dřív se do ní vlepil `last_err`, tj. čtyřicetiřádkový JSON od Gemini — přesně to
    JP 2026-08-15 dostal do Open WebUI. Chybová hláška je taky produkt.
    """

    class DeadAvail:
        def available(self, model):
            return False, "429→54s"
        def record_use(self, model):
            pass
        def record_429(self, model, err=None, default_wait=60):
            return default_wait

    def setUp(self):
        self._tiers = R.TIER_CANDIDATES
        self._decide = R.decide_tier
        R.TIER_CANDIDATES = {k: ["gemini/gemini-flash-lite-latest", "cli:claude:sonnet"]
                             for k in self._tiers}
        R.decide_tier = lambda m, e, i: ("stredni", "[test]")

    def tearDown(self):
        R.TIER_CANDIDATES = self._tiers
        R.decide_tier = self._decide

    def test_message_is_prose_not_json(self):
        _t, used, text, _i = R.answer([{"role": "user", "content": "ahoj"}],
                                      None, None, self.DeadAvail())
        self.assertIsNone(used)
        self.assertNotIn("{", text, "do chatu nesmí odejít JSON")
        self.assertNotIn("litellm", text)
        self.assertIn("Zkus to znovu za ~54 s", text)

    def test_explains_why_claude_was_not_used_when_tools_present(self):
        """Bez tohohle vypadá 'požádal jsem o Claude a nepoužil se' jako výpadek Claude."""
        _t, _u, text, _i = R.answer([{"role": "user", "content": "ahoj"}],
                                    None, None, self.DeadAvail(),
                                    tools=[{"type": "function",
                                            "function": {"name": "terminal"}}])
        self.assertIn("nástroje", text)
        self.assertNotIn("{", text)

    def test_no_tools_does_not_mention_claude_reason(self):
        _t, _u, text, _i = R.answer([{"role": "user", "content": "ahoj"}],
                                    None, None, self.DeadAvail())
        self.assertNotIn("neumí přenést", text)


class CandidateFiltering(unittest.TestCase):
    """answer() must never hand a tools request to a candidate that would drop them."""

    def test_filter_keeps_only_capable(self):
        tier = ["cli:claude:opus", "gemini/gemini-flash-latest", "remote:codex",
                "gemini/gemini-flash-lite-latest"]
        kept = [m for m in tier if R.supports_tools(m)]
        self.assertEqual(kept, ["gemini/gemini-flash-latest",
                                "gemini/gemini-flash-lite-latest"])

    def test_tier_with_no_capable_candidate_is_detectable(self):
        # "tezky" leads with Claude/Codex; a tools request there must fail loudly rather
        # than quietly answering without tools.
        tier = ["cli:claude:opus", "cli:claude:sonnet", "remote:codex"]
        self.assertEqual([m for m in tier if R.supports_tools(m)], [])

    def test_every_real_tier_has_a_capable_candidate_after_fallthrough(self):
        """The tier map is advisory for tool requests — measured on the live map (A-070),
        `tezky` filters down to ONE capable candidate, and it is the RPD-20 row that is
        spent most days. Fall-through must give every tier more than that single option."""
        all_capable = []
        for models in R.TIER_CANDIDATES.values():
            for m in models:
                if m not in all_capable and R.supports_tools(m):
                    all_capable.append(m)
        self.assertGreater(len(all_capable), 1,
                           "fall-through pool must not collapse to one model")
        for tier, models in R.TIER_CANDIDATES.items():
            own = [m for m in models if R.supports_tools(m)]
            merged = own + [m for m in all_capable if m not in own]
            self.assertTrue(merged, f"tier {tier} has no capable candidate at all")
            self.assertGreaterEqual(
                len(merged), 2,
                f"tier {tier} would dead-end on a single model: {merged}")
            if own:
                self.assertEqual(merged[0], own[0],
                                 f"tier {tier} must still prefer its own first choice")


class ServerEnvelope(unittest.TestCase):
    """The finish_reason / message branch in server.py, without starting an HTTP server."""

    @staticmethod
    def envelope(result):
        if isinstance(result, dict):
            message, finish = result, ("tool_calls" if result.get("tool_calls") else "stop")
        else:
            message, finish = {"role": "assistant", "content": result}, "stop"
        return message, finish

    def test_plain_text_result(self):
        message, finish = self.envelope("ahoj")
        self.assertEqual(finish, "stop")
        self.assertEqual(message, {"role": "assistant", "content": "ahoj"})

    def test_tool_call_result_sets_finish_reason(self):
        result = {"role": "assistant", "content": None,
                  "tool_calls": [{"id": REAL_ID, "type": "function",
                                  "function": {"name": "f", "arguments": "{}"}}]}
        message, finish = self.envelope(result)
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(message["tool_calls"][0]["id"], REAL_ID)

    def test_dict_result_without_calls_is_still_stop(self):
        message, finish = self.envelope({"role": "assistant", "content": "done"})
        self.assertEqual(finish, "stop")


class TestMsgText(unittest.TestCase):
    """`content` není vždycky str. Tohle je regresní test na 500 ze zpátečního legu.

    Historie: první leg tool callu (dotaz -> tool_calls) fungoval a byl otestovaný,
    takže Shape 1 vypadal hotově. Druhý leg — poslat assistant zprávu s tool callem
    a výsledek nástroje zpátky — shazoval router VŽDY, protože ta assistant zpráva
    má `content: None`. Chyba padla před routováním, takže v journalu nebyla.
    """

    # Přesně to, co vrátí OpenAI-kompatibilní klient zpátky na druhém legu.
    TOOL_LEG = [
        {"role": "user", "content": "kolik je místa?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "df", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"free_gb": 412}'},
    ]

    def test_none_content_is_empty_string(self):
        self.assertEqual(R.msg_text({"role": "assistant", "content": None}), "")

    def test_missing_content_key(self):
        self.assertEqual(R.msg_text({"role": "assistant"}), "")

    def test_multimodal_list_returns_text_parts_only(self):
        m = {"role": "user", "content": [
            {"type": "text", "text": "co je na obrázku"},
            {"type": "image_url", "image_url": {"url": "data:..."}}]}
        self.assertEqual(R.msg_text(m), "co je na obrázku")

    def test_char_count_over_tool_leg_does_not_raise(self):
        """Přesně ten výraz, co padal — jen v serveru, ne tady."""
        self.assertEqual(sum(len(R.msg_text(m)) for m in self.TOOL_LEG), 14 + 0 + 17)

    def test_flatten_survives_tool_leg_and_labels_the_tool(self):
        out = R._flatten(self.TOOL_LEG)
        self.assertIn("Nástroj: ", out)         # výsledek nástroje není řeč asistenta
        self.assertNotIn("None", out)

    def test_decide_tier_survives_tool_leg(self):
        """Klasifikátor bere jen user zprávy, ale historii dostane celou."""
        got = [R.msg_text(m) for m in self.TOOL_LEG if m.get("role") == "user"]
        self.assertEqual(got, ["kolik je místa?"])

    def test_parse_override_on_none_content_does_not_raise(self):
        msgs = [{"role": "user", "content": None}]
        self.assertEqual(R.parse_override(msgs), (None, msgs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
