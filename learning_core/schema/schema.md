# Distillation schema — JP profile

This file governs how the distiller turns raw conversations into the wiki profile.
It is the ONLY place the distillation behavior is defined. JP evolves it over time.

## Goal
Maintain `wiki/profile.md` — a concise, accurate, living profile of JP, distilled
from his conversations with his assistant. It is what makes the assistant "know JP".

## Hard rules (the distiller MUST obey)
1. **Only supported facts.** Every statement must be grounded in the raw conversations
   or the existing profile. Never invent, guess, or extrapolate personality.
1a. **NEVER make a fact MORE SPECIFIC than the evidence.** Do not turn a place into a
   venue, a topic into an activity, an interest into a habit, or a mention into a routine.
   If the source says JP lives near X, the profile says JP lives near X — not that he
   visits anything at X. When tempted to enrich a detail, write the plainer version.
   **This rule exists because rule 1 was not enough.** On 2026-08-10 the profile carried
   *"active near Hrnčířská street in Ponava"* — true, JP's district. A distiller pass
   rewrote it as *"active near Hrnčířská street in Ponava, **visits the Ponávka indoor
   pool**"*. Ponava is a district; Ponávka is a swimming pool; they sound alike and are
   both in Brno. JP had never heard of the pool. Nothing in the corpus mentioned swimming.
   The profile is rewritten COMPLETELY every run, so each pass is another chance to
   embellish — and an embellishment, once written, becomes "the existing profile" and
   therefore looks like grounding on the next pass.
1b. **The corpus contains BOTH sides of every conversation. Only JP's own statements
   about himself are facts about him.** An assistant suggestion, guess, or recommendation
   is NOT evidence — not even when JP replies enthusiastically. Agreement with a guess is
   not confirmation of a fact; people say "good idea!" to things they have never done.
   In the Ponávka case the assistant suggested the pool (from its own fabricated profile
   line), JP replied *"actually to byl dost dobrý recommendation :D hustý hustý, jsem
   nevěděl, že je to bazén v Brně"* — which literally says he did not know the place — and
   that exchange is now in the corpus, reading like corroboration. **If the only support
   for a claim is the assistant having said it first, the claim does not go in the profile.**
1c. **When a profile line carries a correction note, honour it.** Lines marked with a
   dated `NOTE:` recording that something was false are JP's ground truth. Keep the note,
   keep the corrected fact, and do not re-introduce what it retracts.
2. **Durable over transient.** Record stable facts (who JP is, projects, preferences,
   how he works). Skip one-off chatter and momentary states.
3. **Update in place, don't append endlessly.** Merge new info into the right section;
   refine wording; remove things later contradicted or retracted.
4. **Contradictions are surfaced, not silently resolved.** If new info conflicts with
   the existing profile, keep BOTH under `## Contradictions to resolve` with dates,
   rather than overwriting. JP resolves them.
5. **Concise.** The whole profile should stay readable in one sitting (aim < ~400 lines).
   Prefer tight bullet points over prose. Cut redundancy every run.
6. **Language: English**, because the assistant reasons in English. BUT preserve
   verbatim Czech phrases when *how* JP says something matters (quote them).
7. **No sensitive secrets** in the profile (API keys, passwords, precise location).
   If raw contains them, do not copy them in.
8. **Transparency:** the profile must always read as something JP could verify against
   his own conversations. No opaque "the model thinks" claims.

## Fixed sections (keep this order; omit a section only if truly empty)
- `## Identity & context` — who JP is, role, situation, language (CZ speaker, mixes EN).
- `## Active projects & goals` — what he's building / working toward, with status.
- `## Working style & preferences` — how he likes to work, communicate, decide.
- `## Technical environment` — tools, machines, stack he uses.
- `## Interests` — durable interests, hobbies, communities, and topics JP cares about
  (e.g. FOSS, privacy, Bitcoin/Monero, self-hosting, festivals, liberty group). Keep it to
  stable interests, not passing curiosities.
- `## Recurring themes` — patterns across conversations (e.g. "wants everything free").
- `## Open threads` — unfinished things to follow up.
- `## Contradictions to resolve` — conflicts awaiting JP (see rule 4). Omit if none.

## Output
Return the COMPLETE updated `profile.md` as markdown. No preamble, no code fences.
