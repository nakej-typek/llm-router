"""profile_history.py — permanent, append-only record of what each distiller run changed.

STATUS: NOT WIRED IN. Nothing imports this. Written by WIN-CLAUDE 2026-08-11 (W-060/W-066)
for ARCH to review and hook up; `distiller.py` executes live from the Syncthing tree every
30 minutes, so the two-line call site is ARCH's to add, not mine.

WHY THIS EXISTS
---------------
On 2026-08-10 a distiller pass turned "lives near Hrnčířská street in Ponava" (a district)
into "visits the Ponávka indoor pool" (a swimming pool). Nothing in the corpus supported it.
`persona.py` then injected the profile into every router call, the model recommended the pool,
JP was pleased, and his pleased reply was captured back into `raw/` — where the next pass
reads it as corroboration. Fabrication plus a feedback loop.

It was traceable only by luck: a 2026-07-26 baseline happened to still exist. It would not
have been. `backup_profile()` keeps `KEEP_BACKUPS = 5` snapshots against a 30-minute timer,
so the record of when a claim entered the profile is roughly **two and a half hours deep**.

That matters more than it sounds, because of how the fabrication defeats the obvious audit:
once a claim has looped once, the corpus *does* mention it. Grepping raw/ for support clears
exactly the claims that have already been echoed back. The only question that separates a
fabrication from a fact is **"did the corpus say this BEFORE the profile did?"** — and
answering it needs the profile's history, which is currently being deleted on a rolling basis.

SIZE, measured against the real backups (W-067 measured 1,847 B and was wrong — that was the
middle of three transitions; corrected in A-073 and re-verified independently):

    unchanged run                  0 B      the log only grows when the profile moves
    ordinary changed run       1,847 B
    heavy distillation         9,470 B      tonight's — and nights like this are exactly
                                            when provenance matters, so the expensive case
                                            and the important case coincide
    average over 3 transitions 3,772 B   ->  ~63 MB/year worst case (A-073), i.e. if every
                                            one of the 48 daily fires changed something.
                                            1 of 3 sampled did not. Three data points is not
                                            enough for a confident middle estimate.

GROWTH IS THE REAL COST, NOT DISK. This file lives in `.backups/`, inside the Syncthing tree,
and is append-only by design — so it replicates between both machines forever. That is a
different concern from 63 MB and it is the one that quietly becomes a problem. It is kept
synced deliberately: the whole reason it exists is that evidence vanished, and a machine-local
log dies with the machine and cannot be read from Windows at all. `LOG_YEARLY` bounds any
single file so old years can be archived out of the sync set without touching the current one.

HOW TO WIRE IT (ARCH)
---------------------
In `distiller.py`, `write_profile()` is where old and new both exist. Add:

    import profile_history                                    # near the other imports

    def write_profile(text):
        old = ""
        if os.path.exists(WIKI):
            with open(WIKI, encoding="utf-8") as f:
                old = f.read()
        profile_history.append(WIKI, old, text)               # <- before the replace
        tmp = WIKI + ".tmp"
        ...

Failure here must never block a distillation — the log is an audit aid, not a critical path —
so `append()` swallows its own exceptions and says so below rather than relying on the caller
to remember a try/except.
"""
import difflib
import os
from datetime import datetime

LOG_NAME = "profile.history.log"

# Roll over yearly (A-073). Not rotation — nothing is ever deleted, which is the entire point.
# It bounds any single file so a past year can be archived out of the Syncthing set later
# without touching the current one. Set False for one unbroken file.
LOG_YEARLY = True


def log_path(wiki_path, when=None):
    name = LOG_NAME
    if LOG_YEARLY:
        year = (when or datetime.now()).year
        name = f"profile.history.{year}.log"
    return os.path.join(os.path.dirname(wiki_path), ".backups", name)


def render(old_text, new_text, stamp=None, context=1):
    """Unified diff of one distillation, or "" when nothing changed.

    context=1 rather than 0: a bare changed line is often unreadable months later, and one
    line either side is usually enough to see which section it landed in. Still tiny.
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines,
                                     fromfile="before", tofile="after", n=context))
    if not diff:
        return ""
    stamp = stamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    added = sum(1 for d in diff if d.startswith("+") and not d.startswith("+++"))
    removed = sum(1 for d in diff if d.startswith("-") and not d.startswith("---"))
    head = f"\n===== {stamp} | +{added} -{removed} lines =====\n"
    return head + "".join(diff if diff[-1].endswith("\n") else diff + ["\n"])


def append(wiki_path, old_text, new_text, stamp=None):
    """Append this run's diff. Returns True if something was written.

    Never raises: a distillation must not fail because its audit log could not be written.
    An unwritten log line loses provenance for one run; a raised exception here would lose
    the distillation itself, which is strictly worse.
    """
    try:
        entry = render(old_text, new_text, stamp=stamp)
        if not entry:
            return False
        path = log_path(wiki_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Self-test on the real Ponávka case, so the thing this exists to catch is the thing
    # it is demonstrated against. Runs anywhere, touches nothing.
    before = ("- Based in Brno (active near Hrnčířská street in Ponava), "
              "relocating to Prague.\n- Works in C#.\n")
    after = ("- Based in Brno (active near Hrnčířská street in Ponava, visits the "
             "Ponávka indoor pool), relocating to Prague.\n- Works in C#.\n")
    out = render(before, after, stamp="2026-08-10 18:35:19")
    assert "Ponávka" in out and out.count("\n") < 12, out
    assert render(before, before) == "", "unchanged run must log nothing"
    # The corpus is Czech; a Windows console is cp1252 and raises on it. Write bytes with
    # replacement rather than letting a self-test fail over console encoding — the diff
    # itself is written to a file as UTF-8 and is unaffected.
    import sys
    sys.stdout.buffer.write(out.encode(sys.stdout.encoding or "utf-8", errors="replace"))
    print("\nOK - a fabrication of this shape leaves one greppable line, permanently.")
