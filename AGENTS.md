# AGENTS.md — Demake Engine

Guidance for AI agents (and humans) working in this repo.

## Project

Turn a genre pick (or a game trailer MP4) into a playable 8-bit browser demake.
FastAPI backend + Phaser.js frontend, all game code currently inline in
`frontend/game.html`. Roadmap and sprint plan live in
`demake-engine-ARCHITECTURE.md` (Section 12 = Sprint Plan — source of truth).

## Sprint Workflow (required)

1. **Before starting a sprint:** its Plan & Scope must be written as a checklist in
   `demake-engine-ARCHITECTURE.md` Section 12. If it isn't there yet, write it first
   and get user sign-off before coding.
2. **While working:** tick off checklist items as they are completed. Keep the sprint's
   status marker current (`⬜` not started / 🔶 in progress / ✅ complete).
3. **After finishing a sprint:**
   - Update status markers in the doc.
   - Commit everything belonging to the sprint with a message like
     `Sprint 8A: shared game systems (entities, inventory, projectiles)`.
   - Push to origin.

Never commit unrelated files. Runtime artifacts (`demake.db`, `__pycache__/`,
`backend/outputs/`, `backend/uploads/`) are untracked — keep them that way.

## Verification expectations

- Python changes: run the backend once (`venv/Scripts/python -m uvicorn main:app --app-dir backend`)
  and hit the affected endpoint before claiming done.
- Frontend changes: syntax-check the JS inside `frontend/game.html`
  (extract the `<script>` body and run `node --check`), then run the headless
  smoke test (`tools/smoke.js` — see usage comment at top of file) for every
  affected template: `node smoke.js <template_id>` must report the correct
  active scene and `NO CONSOLE ERRORS`. Screenshots land in
  `%TEMP%\opencode\smoke\`.
- No test framework is configured; smoke harness + manual verification are the bar.

## Environment notes

- Windows, PowerShell 5.1. Use `cmd1; if ($?) { cmd2 }`, not `&&`.
- Python venv at `venv/`; activate via `venv\Scripts\Activate.ps1`.
- Console is cp1252 — keep prints ASCII-safe in backend code.
- Godot 4 is installed locally for Sprint 9a export work.

## Current position

Sprints 0–7 complete. Sprint 8 series (full gameplay loops) in progress — see the
status table in Section 12 of the architecture doc for per-genre state.
