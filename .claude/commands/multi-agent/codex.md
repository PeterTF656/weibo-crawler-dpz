---
description: Delegate a task to Codex SKILL (fire-and-forget)
argument-hint: [task for Codex]
---

## Codex Delegation (Fire-and-Forget)

Use **Codex SKILL** (`codex`) for executing this task in the background. Make sure more than enough context is passed to it. Fire-and-forget: do not supervise interactively or ask follow-ups unless the task is blocked.

### Instructions

- Treat `$ARGUMENTS` as the full task to delegate to Codex.
- Prefer action over questions; only ask the user if required to proceed safely/correctly.
- Ensure Codex has sufficient context: relevant file/dir pointers, constraints (what can/can’t change), expected output, and how to verify (tests/commands).
- Let Codex run autonomously to completion; then summarize results and next steps (including how to resume).

### Task

$ARGUMENTS
