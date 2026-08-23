"""Testy pro purge_router_sessions — hlavně přepočet offsetů.

Proč zrovna offsety: driver i distiller si drží pozice v bajtech. Když se soubor
zkrátí a offsety se nepřepočítají, ukazují doprostřed cizího textu a obě služby
buď přeskočí kus korpusu, nebo ho zpracují dvakrát — a v obou případech to vypadá
jako "model se choval divně", ne jako chyba tady.

Běž: python3 test_purge.py
"""
import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "purge", str(Path(__file__).parent / "purge_router_sessions.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)


def sec(path, body, sid="1798c679", when="2026-08-14 02:50"):
    return f"## {path} · {sid} · arch · {when}\n{body}\n"


GOOD = "/home/user/syncthing/archlinux"
BAD = "/home/user//local/share/ai/router/router"


class RemapTests(unittest.TestCase):
    def test_offset_before_any_cut_is_unchanged(self):
        self.assertEqual(P.remap(50, [(100, 200)]), 50)

    def test_offset_after_a_cut_shifts_by_its_size(self):
        self.assertEqual(P.remap(300, [(100, 200)]), 200)

    def test_offset_inside_a_cut_falls_to_its_start(self):
        """Kdo měl přečteno doprostřed vyříznutého úseku, má po úklidu přečteno
        právě k jeho začátku — ne dál, aby se nepřeskočil text, který zůstal."""
        self.assertEqual(P.remap(150, [(100, 200)]), 100)

    def test_boundaries(self):
        self.assertEqual(P.remap(100, [(100, 200)]), 100)   # přesně na začátku
        self.assertEqual(P.remap(200, [(100, 200)]), 100)   # přesně na konci

    def test_multiple_cuts_accumulate(self):
        cuts = [(100, 200), (400, 450)]
        self.assertEqual(P.remap(90, cuts), 90)
        self.assertEqual(P.remap(300, cuts), 200)
        self.assertEqual(P.remap(500, cuts), 350)

    def test_remap_matches_a_real_rewrite(self):
        """Kontrola proti skutečnosti, ne proti mé aritmetice: pro každý offset musí
        text OD něj začínat stejně před i po vyříznutí."""
        text = (sec(GOOD, "JP: první\nAsistent: ok")
                + sec(BAD, "JP: Uživatel: idempotence\nAsistent: dlouhá odpověď")
                + sec(GOOD, "JP: druhá\nAsistent: ok")
                + sec(BAD, "JP: Uživatel: další\nAsistent: text")
                + sec(GOOD, "JP: třetí"))
        cuts, _n, _s = P.plan(text)
        keep, prev = [], 0
        for s, e in cuts:
            keep.append(text[prev:s]); prev = e
        keep.append(text[prev:])
        new = "".join(keep)

        for off in range(0, len(text), 7):
            if any(s <= off < e for s, e in cuts):
                continue                      # uvnitř vyříznutého úseku se text nezachová
            # Porovnávat jen PO nejbližší řez. Za ním text logicky pokračovat stejně
            # nemůže — tam právě něco zmizelo. (První verze tohohle testu četla 40 znaků
            # napevno, přetekla přes hranici a hlásila chybu, která žádná nebyla.)
            nxt = min((s for s, _e in cuts if s > off), default=len(text))
            n = nxt - off
            self.assertEqual(new[P.remap(off, cuts):][:n], text[off:nxt],
                             f"offset {off} ukazuje po přepočtu jinam")


class PlanTests(unittest.TestCase):
    def test_only_router_sections_are_cut(self):
        text = sec(GOOD, "JP: a") + sec(BAD, "JP: b") + sec(GOOD, "JP: c")
        cuts, chars, secs = P.plan(text)
        self.assertEqual(secs, 1)
        cut_text = text[cuts[0][0]:cuts[0][1]]
        self.assertIn("JP: b", cut_text)
        self.assertNotIn("JP: a", cut_text)
        self.assertNotIn("JP: c", cut_text)
        self.assertEqual(chars, len(cut_text))

    def test_markdown_headings_inside_a_conversation_do_not_split(self):
        """V přepisech jsou běžné nadpisy ('## Stav', '## Active projects & goals').
        Kdyby je HEADER bral jako hranici sekce, vyřízlo by se něco jiného, než mělo —
        a to je přesně ta chyba, kterou by nikdo nezpozoroval."""
        text = sec(BAD, "JP: x\n## Stav\n## Active projects & goals\nAsistent: y") \
            + sec(GOOD, "JP: zůstat")
        cuts, _c, secs = P.plan(text)
        self.assertEqual(secs, 1)
        self.assertIn("JP: zůstat", text[cuts[0][1]:])
        self.assertNotIn("JP: zůstat", text[cuts[0][0]:cuts[0][1]])

    def test_clean_corpus_yields_no_cuts(self):
        text = sec(GOOD, "JP: a") + sec(GOOD, "JP: b")
        self.assertEqual(P.plan(text), ([], 0, 0))

    def test_trailing_bad_section_runs_to_end_of_file(self):
        text = sec(GOOD, "JP: a") + sec(BAD, "JP: konec")
        cuts, _c, _s = P.plan(text)
        self.assertEqual(cuts[0][1], len(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
