"""Fáze 2 — per-chat Claude session reuse (claude_sdk_backend._plan/_remember).

These are pure-function tests: no SDK, no network, no CLI. What they protect is the
decision of WHETHER to resume, which is the only part that can silently corrupt a
conversation — a wrong resume splices two people's chats together, and it would look
like the model hallucinating rather than like a bug here.

Each test says what a wrong implementation would do, because a resume test that only
ever asserts "returns None" passes against code that never resumes at all. Where a test
asserts a fallback, it also asserts a control case that DOES resume.

Run:  python3 test_session.py
"""
import sys
import types
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Same stub as test_tools.py — see the comment there. Keep the name list in sync with
# what claude_sdk_backend.py imports, or this file dies at import and stops testing.
_stub = types.ModuleType("claude_agent_sdk")
for _name in ("ClaudeSDKClient", "ClaudeAgentOptions", "AssistantMessage", "TextBlock",
              "StreamEvent", "ResultMessage"):
    setattr(_stub, _name, type(_name, (), {}))
sys.modules.setdefault("claude_agent_sdk", _stub)

import claude_sdk_backend as B  # noqa: E402

SYS = "persona"
MODEL = "sonnet"


def _msgs(*pairs):
    """('u','ahoj'), ('a','čau') -> OpenAI-shaped message list."""
    role = {"u": "user", "a": "assistant"}
    return [{"role": role[r], "content": c} for r, c in pairs]


class SessionPlanTests(unittest.TestCase):
    def setUp(self):
        B._SESS.clear()

    def test_first_turn_is_cold_and_sends_everything(self):
        resume, turns, _key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        self.assertIsNone(resume)
        self.assertEqual([t["content"] for t in turns], ["ahoj"])
        self.assertEqual(len(digests), 1)

    def test_second_turn_resumes_and_sends_only_the_new_user_message(self):
        """The whole point of fáze 2, and the trap in it.

        The resumed CLI session ALREADY holds the answer it generated, so the reply must
        be counted as sent — otherwise turn 2 re-sends "čau" as if the user had typed it
        and the model answers its own words. `_remember(answer=…)` is what prevents that;
        drop the answer= argument and this test fails on the content assertion while
        `resume is not None` still passes."""
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-abc", digests, answer="čau")

        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "co je router"))
        resume, turns, key2, _d = B._plan(MODEL, SYS, m2)
        self.assertEqual(resume, "sid-abc")
        self.assertEqual(key2, key)
        self.assertEqual([t["content"] for t in turns], ["co je router"])

    def test_third_turn_keeps_resuming(self):
        """Two turns can work by accident; the offset has to keep advancing."""
        _r, _t, key, d1 = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-1", d1, answer="čau")

        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "druhý"))
        _r2, _t2, key2, d2 = B._plan(MODEL, SYS, m2)
        B._remember(key2, "sid-1", d2, answer="druhá odpověď")

        m3 = m2 + _msgs(("a", "druhá odpověď"), ("u", "třetí"))
        resume, turns, _k, _d = B._plan(MODEL, SYS, m3)
        self.assertEqual(resume, "sid-1")
        self.assertEqual([t["content"] for t in turns], ["třetí"])

    def test_client_altered_answer_falls_back_instead_of_desyncing(self):
        """If a client normalises the assistant text, the stored prefix stops matching.
        That must cost a cold call, never a silently divergent session."""
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-q", digests, answer="čau  ")     # trailing spaces stored

        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "dál"))  # client trimmed them
        resume, turns, _k, _d = B._plan(MODEL, SYS, m2)
        self.assertIsNone(resume)
        self.assertEqual(len(turns), 3)

    def test_identical_opener_in_a_different_chat_does_not_resume(self):
        """JP opens chats with 'ahoj'. Keying on the first message alone would splice the
        second 'ahoj' chat onto the first one's session — the exact context leak that the
        old 'never reuse a session' rule existed to prevent."""
        first = _msgs(("u", "ahoj"), ("a", "odpověď A"))
        _r, _t, key, digests = B._plan(MODEL, SYS, first)
        B._remember(key, "sid-a", digests)

        other = _msgs(("u", "ahoj"), ("a", "úplně jiná odpověď"), ("u", "další dotaz"))
        resume, turns, _k, _d = B._plan(MODEL, SYS, other)
        self.assertIsNone(resume)
        self.assertEqual(len(turns), 3)         # falls back to the full transcript

        # control: the chat it really belongs to still resumes
        same = first + _msgs(("u", "pokračuj"))
        self.assertEqual(B._plan(MODEL, SYS, same)[0], "sid-a")

    def test_edited_history_does_not_resume(self):
        """OWUI regenerate/edit rewrites an earlier turn. Resuming would answer against
        the version the user just replaced."""
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj"), ("a", "čau")))
        B._remember(key, "sid-x", digests)

        edited = _msgs(("u", "ahoj"), ("a", "PŘEPSANÁ odpověď"), ("u", "pokračuj"))
        resume, turns, _k, _d = B._plan(MODEL, SYS, edited)
        self.assertIsNone(resume)
        self.assertEqual(len(turns), 3)

    def test_retry_of_the_same_turn_is_cold(self):
        """Nothing new to send. Resuming here would query the CLI with an empty prompt."""
        m = _msgs(("u", "ahoj"), ("a", "čau"))
        _r, _t, key, digests = B._plan(MODEL, SYS, m)
        B._remember(key, "sid-y", digests)

        resume, turns, _k, _d = B._plan(MODEL, SYS, m)
        self.assertIsNone(resume)
        self.assertEqual(len(turns), 2)

    def test_expired_session_is_cold(self):
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-old", digests)
        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "dál"))
        self.assertEqual(B._plan(MODEL, SYS, m2)[0], "sid-old")   # control: alive

        B._SESS[key].ts = time.time() - B.SESSION_TTL_SEC - 1
        self.assertIsNone(B._plan(MODEL, SYS, m2)[0])

    def test_model_is_part_of_the_key(self):
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-s", digests)

        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "dál"))
        self.assertIsNone(B._plan("opus", SYS, m2)[0])            # other model
        self.assertEqual(B._plan(MODEL, SYS, m2)[0], "sid-s")     # control

    def test_a_changed_system_prompt_still_resumes(self):
        """router.answer() rebuilds the system prompt per call and stamps the tier, the
        model label and the rotation note into it. Those move between turns of ONE chat,
        so a key that includes them drops the session — which is what actually happened
        on 2026-08-14 before the key was narrowed. This is the regression guard."""
        _r, _t, key, digests = B._plan(MODEL, "persona [tier=stredni]",
                                       _msgs(("u", "ahoj")))
        B._remember(key, "sid-t", digests, answer="čau")

        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "dál"))
        resume, turns, _k, _d = B._plan(
            MODEL, "persona [tier=tezky] [rotace: flash-lite(cooldown 41s)]", m2)
        self.assertEqual(resume, "sid-t")
        self.assertEqual([t["content"] for t in turns], ["dál"])

    def test_forget_drops_the_entry(self):
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-z", digests)
        m2 = _msgs(("u", "ahoj"), ("a", "čau"), ("u", "dál"))
        self.assertEqual(B._plan(MODEL, SYS, m2)[0], "sid-z")     # control

        B._forget(key)
        self.assertIsNone(B._plan(MODEL, SYS, m2)[0])

    def test_remember_ignores_a_missing_session_id(self):
        """A call that never yielded a ResultMessage has no id. Storing None would make
        the next turn resume with resume=None — a cold call recorded as a warm one."""
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, None, digests)
        self.assertNotIn(key, B._SESS)

    def test_empty_answer_still_advances_the_prefix(self):
        """A turn the model answered with nothing was still SENT, so the session holds
        it. Forgetting that froze the stored prefix and made every later turn re-send
        more of the transcript (1 → 3 → 5 → 7 messages, seen live 2026-08-14)."""
        _r, _t, key, digests = B._plan(MODEL, SYS, _msgs(("u", "ahoj")))
        B._remember(key, "sid-e", digests, answer="")

        m2 = _msgs(("u", "ahoj"), ("a", ""), ("u", "dál"))
        resume, turns, _k, _d = B._plan(MODEL, SYS, m2)
        self.assertEqual(resume, "sid-e")
        self.assertEqual([t["content"] for t in turns], ["dál"])

    def test_registry_is_bounded(self):
        for i in range(B.SESSION_MAX + 25):
            B._remember(f"k{i}", f"sid{i}", [B._digest(str(i))])
        self.assertLessEqual(len(B._SESS), B.SESSION_MAX)

    def test_system_messages_are_not_part_of_the_transcript(self):
        """system_prompt goes through the SDK's own field; counting a system message as a
        turn would shift the prefix by one and make every second turn fall back to cold."""
        m = [{"role": "system", "content": "persona"}] + _msgs(("u", "ahoj"))
        _r, turns, key, digests = B._plan(MODEL, SYS, m)
        self.assertEqual([t["content"] for t in turns], ["ahoj"])
        B._remember(key, "sid-sys", digests, answer="čau")

        m2 = [{"role": "system", "content": "persona"}] + _msgs(
            ("u", "ahoj"), ("a", "čau"), ("u", "dál"))
        resume, turns2, _k, _d = B._plan(MODEL, SYS, m2)
        self.assertEqual(resume, "sid-sys")
        self.assertEqual([t["content"] for t in turns2], ["dál"])

    def test_none_content_does_not_raise(self):
        """A tool leg can carry content=None; router.py has the same guard."""
        m = [{"role": "user", "content": None}, {"role": "assistant", "content": None}]
        resume, turns, _k, digests = B._plan(MODEL, SYS, m)
        self.assertIsNone(resume)
        self.assertEqual(len(digests), 2)
        self.assertEqual(len(turns), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
