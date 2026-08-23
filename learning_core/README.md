# Learning core — Karpathy 3-layer corpus

Makes the assistant progressively *know the owner*. A readable markdown corpus is the single
source of truth; Hermes feeds it (raw) and reads it (context). Free (Gemini flash-lite),
stdlib only, no deps. See CONSTITUTION.md (Articles B & D) and KARPATHY.md.

## Layers
```
raw/      append-only conversations           ← what happened
wiki/     distilled living corpus (profile.md) ← what was learned  (SOURCE OF TRUTH)
schema/   distillation rules (schema.md)       ← the owner evolves this
distiller.py  reads new raw → updates wiki per schema (free Gemini flash-lite)
```

## Status
- ✅ distiller.py — built & tested end-to-end (produces an accurate profile in ~1s).
- ✅ schema.md, wiki/profile.md skeleton, raw/ format.
- ✅ distiller.service + distiller.timer (systemd --user, 30-min cadence).
- ⏳ RAW CAPTURE — feeding Hermes↔the owner Signal exchanges into raw/ (pending recon W-011).
- ⏳ WIKI INJECTION — Hermes reading profile.md as context (pending recon W-011).

## Deploy (on Arch, no sudo) — do AFTER the raw-capture wiring is designed
```bash
cp distiller.service distiller.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now distiller.timer
systemctl --user list-timers | grep distiller     # confirm scheduled
# manual one-off pass:
python3 ~/syncthing/archlinux/ai_router/learning_core/distiller.py
```

## How it behaves
- Idempotent: tracks per-file byte offsets in `.distiller_state.json`; only new raw is processed.
- No-ops if <200 new chars (batches up) → idle runs cost no quota.
- Backs up `wiki/profile.md` before each rewrite (keeps last 5); refuses to overwrite on
  suspicious/empty model output.
- On Gemini 429: skips the run, offsets unchanged, retries next timer.

## The wiki lives in Syncthing on purpose
So the updated profile syncs back to the owner's desktop and can be opened/edited in Obsidian
(transparency — the owner always sees what was learned, and can correct it by hand).
