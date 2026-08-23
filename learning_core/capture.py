#!/usr/bin/env python3
"""
Learning core — raw capture (Route B: full-fidelity SQLite export).

Reads Hermes's session store (~/.hermes/state.db) and appends new JP<->Hermes
Signal exchanges to raw/hermes-signal.md (append-only, the distiller's input).
Full fidelity: message content in state.db is UNtruncated (unlike the agent:end
hook, which caps text at 500 chars).

Scope / privacy (Article 1): local only, no cloud. Only Signal traffic is read,
and thanks to the SIGNAL_GROUP_ONLY patch (W-009) the agent only ever processes /
stores messages from the dedicated "Hermes" group — DMs / Note-to-Self are dropped
at intake and never reach the DB. state.db carries no chat_id column, so group-vs-DM
can't be filtered here; instead the FIRST run baselines at the current max message id,
so pre-W-009 history (which may contain Note-to-Self) is never ingested. Everything
captured afterwards is, by construction, the Hermes group only.

Idempotent: tracks the last exported messages.id in .capture_state.json; re-runs never
duplicate. A trailing in-flight user turn (no assistant yet) is left for the next run.
stdlib only, no deps.
"""

import json
import os
import sqlite3
import sys
from datetime import datetime

BASE = os.environ.get("LEARNING_CORE_DIR", os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "raw", "hermes-signal.md")
STATE = os.path.join(BASE, ".capture_state.json")
STATE_DB = os.path.expanduser("~/.hermes/state.db")

SOURCE = "signal"
# Assistant messages matching these are transport errors, not real answers — skip the turn.
ERROR_PREFIXES = ("API call failed", "❌", "⚠️")


def log(msg):
    print(f"[capture {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state():
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE)


def connect_ro():
    # Read-only; safe to run while Hermes writes. Short busy timeout.
    uri = f"file:{STATE_DB}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def session_short(session_id):
    # "20260724_064340_fc7626f6" -> "fc7626f6"
    return str(session_id).rsplit("_", 1)[-1] if session_id else "?"


def is_error(text):
    t = (text or "").lstrip()
    return any(t.startswith(p) for p in ERROR_PREFIXES)


def fmt_turn(user_row, asst_row):
    ts = datetime.fromtimestamp(user_row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S")
    sid = session_short(user_row["session_id"])
    return (
        f"## {ts}  (session {sid})\n"
        f"**JP:** {(user_row['content'] or '').strip()}\n\n"
        f"**Hermes:** {(asst_row['content'] or '').strip()}\n\n"
    )


def main():
    if not os.path.exists(STATE_DB):
        log(f"no state.db at {STATE_DB} — nothing to do.")
        return
    con = connect_ro()

    state = load_state()
    last_id = state.get("last_id")

    max_id = con.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]

    # First run: baseline at current max id so pre-W-009 history (possible Note-to-Self)
    # is never ingested. Capture starts from the NEXT message onward.
    if last_id is None:
        save_state({"last_id": max_id})
        log(f"first run — baselined at id {max_id}; future group messages will be captured.")
        return

    rows = con.execute(
        """
        SELECT m.id, m.session_id, m.role, m.content, m.timestamp
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE s.source = ?
          AND m.id > ?
          AND m.role IN ('user', 'assistant')
          AND m.content IS NOT NULL
        ORDER BY m.id
        """,
        (SOURCE, last_id),
    ).fetchall()

    if not rows:
        log("no new messages.")
        return

    # Pair user -> assistant within the same session. Advance the offset only up to the
    # last assistant seen, so a trailing in-flight user turn is re-read next run.
    turns = []
    pending = None
    last_assistant_id = None
    for m in rows:
        if m["role"] == "user":
            pending = m
        elif m["role"] == "assistant":
            last_assistant_id = m["id"]
            if pending is not None and pending["session_id"] == m["session_id"]:
                if not is_error(m["content"]):
                    turns.append(fmt_turn(pending, m))
            pending = None

    if last_assistant_id is None:
        log(f"{len(rows)} new row(s) but no completed turn yet — leaving offset at {last_id}.")
        return

    if turns:
        new_file = not os.path.exists(RAW)
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        with open(RAW, "a", encoding="utf-8") as f:
            if new_file:
                f.write("# Raw — Hermes ↔ JP (Signal, group-only)\n\n"
                        "Append-only, full-fidelity. Consumed by distiller.py.\n\n")
            f.write("".join(turns))
        log(f"appended {len(turns)} turn(s); offset {last_id} -> {last_assistant_id}.")
    else:
        log(f"{len(rows)} new row(s), 0 usable turns (errors/unpaired); offset -> {last_assistant_id}.")

    save_state({"last_id": last_assistant_id})


if __name__ == "__main__":
    main()
