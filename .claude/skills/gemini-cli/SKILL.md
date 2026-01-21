---
name: gemini-cli
description: Use when the user asks to run Gemini CLI (gemini command, gemini resume) or references Google Gemini for code analysis, generation, refactoring, or AI-assisted tasks. Invokes Gemini CLI via Bash commands.
---

# Gemini CLI Skill

## CRITICAL: Sandbox Requirement

> **⚠️ All `gemini` commands MUST use `dangerouslyDisableSandbox: true`**

Claude Code's sandbox blocks localhost binding, causing Gemini to crash with `EPERM`. The `permissions.allow` list only skips approval prompts, NOT sandbox restrictions.

```json
{
  "command": "gemini -m gemini-3-flash \"prompt\" 2>/dev/null",
  "dangerouslyDisableSandbox": true
}
```

## Models (MANDATORY)

| Model | Use Case |
|-------|----------|
| `gemini-3-flash` | Default - simple/normal tasks, code review |
| `gemini-3-pro-high` | Complex reasoning, architecture, multi-step tasks |

**Only these two models are allowed.**

## Quick Reference

| Task | Command |
|------|---------|
| Normal task | `gemini -m gemini-3-flash "prompt" 2>/dev/null` |
| Complex task | `gemini -m gemini-3-pro-high "prompt" 2>/dev/null` |
| Auto-edit files | `gemini -m gemini-3-flash --approval-mode auto_edit "prompt" 2>/dev/null` |
| Full auto (ask user first) | `gemini -m gemini-3-flash --yolo "prompt" 2>/dev/null` |
| Resume session | `gemini --resume latest "follow-up" 2>/dev/null` |
| Docker sandbox | `gemini -s -m gemini-3-flash "prompt" 2>/dev/null` |

## Notes

- Always append `2>/dev/null` to suppress verbose output
- Before using `--yolo`, ask user permission via `AskUserQuestion`
- After completion, inform user they can resume with `gemini --resume latest`
- On failure, report error and ask for direction before retrying
