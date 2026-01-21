---
name: codex
description: Use when the user asks to run Codex CLI (codex exec, codex resume) or references OpenAI Codex for code analysis, refactoring, or automated editing. Invokes Codex via Bash commands (not MCP).
---

# Codex Skill Guide

This skill invokes the Codex CLI via Bash commands. **Do not use MCP tools for Codex.**

## CRITICAL: Claude Code Sandbox Compatibility

> **⚠️ MANDATORY: All `codex exec` and `codex resume` commands MUST use `dangerouslyDisableSandbox: true` in the Bash tool call.**

Codex CLI accesses macOS System Configuration APIs at startup (for proxy/network detection), which are blocked by Claude Code's default sandbox. Without disabling the sandbox, Codex will crash with exit code 101:

```
thread 'main' panicked at system-configuration-0.6.1/src/dynamic_store.rs:154:1:
Attempted to create a NULL object.
```

**Why this happens:**
- `codex --version` works in sandbox (no system API calls)
- `codex exec` crashes in sandbox (requires SCDynamicStore API access)
- The sandbox blocks low-level macOS system calls, not just filesystem/network

**Important distinctions:**
- `permissions.allow` list (e.g., `Bash(codex:*)`) only skips approval prompts, NOT sandbox
- `--sandbox danger-full-access` is Codex's internal sandbox setting, NOT Claude Code's sandbox
- You MUST set `dangerouslyDisableSandbox: true` on the Bash tool call itself

**Correct Bash tool invocation:**
```json
{
  "command": "echo \"your prompt\" | codex exec --skip-git-repo-check -m gpt-5.2-codex --config model_reasoning_effort=high --sandbox danger-full-access --full-auto 2>/dev/null",
  "description": "Run Codex task",
  "dangerouslyDisableSandbox": true
}
```

## Model Selection (MANDATORY)

| Model | Use Case |
| --- | --- |
| `gpt-5.2` | Planning, analysis, documentation, non-coding tasks |
| `gpt-5.2-codex` | Coding tasks, refactoring, implementation, bug fixes |

**IMPORTANT**: Only `gpt-5.2` and `gpt-5.2-codex` are allowed. No other models.

## Reasoning Effort (MANDATORY)

| Effort | Use Case |
| --- | --- |
| `xhigh` | Complex multi-file changes, architectural decisions, deep analysis |
| `high` | Standard tasks, code review, moderate complexity |

**IMPORTANT**: Only `xhigh` and `high` are allowed. No `medium` or `low`.

## Sandbox Mode (MANDATORY)

**Always use `danger-full-access`** to ensure Codex has full system access and avoids sandbox-related crashes.

## Running a Task

1. **Determine the model** based on task type:
   - Coding task (implementation, refactoring, bug fix) → `gpt-5.2-codex`
   - Non-coding task (planning, analysis, review) → `gpt-5.2`

2. **Select reasoning effort**:
   - Default to `high` for most tasks
   - Use `xhigh` for complex multi-file changes or architectural decisions

3. **Assemble and run the command**:
   ```bash
   echo "your prompt here" | codex exec --skip-git-repo-check \
     -m <gpt-5.2|gpt-5.2-codex> \
     --config model_reasoning_effort=<xhigh|high> \
     --sandbox danger-full-access \
     --full-auto \
     2>/dev/null
   ```

4. **IMPORTANT**: Always append `2>/dev/null` to suppress thinking tokens (stderr). Only show stderr if debugging is needed.

5. **IMPORTANT**: Always use `dangerouslyDisableSandbox: true` in the Bash tool call.

6. Run the command, capture stdout, and summarize the outcome for the user.

7. **After Codex completes**, inform the user: "You can resume this Codex session at any time by saying 'codex resume' or asking me to continue with additional analysis or changes."

### Quick Reference

| Use case | Model | Reasoning | Command template |
| --- | --- | --- | --- |
| Code review or analysis | `gpt-5.2` | `high` | `echo "..." \| codex exec --skip-git-repo-check -m gpt-5.2 --config model_reasoning_effort=high --sandbox danger-full-access --full-auto 2>/dev/null` |
| Complex analysis | `gpt-5.2` | `xhigh` | `echo "..." \| codex exec --skip-git-repo-check -m gpt-5.2 --config model_reasoning_effort=xhigh --sandbox danger-full-access --full-auto 2>/dev/null` |
| Apply local edits | `gpt-5.2-codex` | `high` | `echo "..." \| codex exec --skip-git-repo-check -m gpt-5.2-codex --config model_reasoning_effort=high --sandbox danger-full-access --full-auto 2>/dev/null` |
| Complex refactoring | `gpt-5.2-codex` | `xhigh` | `echo "..." \| codex exec --skip-git-repo-check -m gpt-5.2-codex --config model_reasoning_effort=xhigh --sandbox danger-full-access --full-auto 2>/dev/null` |
| Resume recent session | (inherited) | (inherited) | `echo "prompt" \| codex exec --skip-git-repo-check resume --last 2>/dev/null` |

## Resuming Sessions

When continuing a previous session:
```bash
echo "your follow-up prompt" | codex exec --skip-git-repo-check resume --last 2>/dev/null
```

**Important**: When resuming, do NOT use configuration flags (model, reasoning effort, sandbox) unless explicitly requested by the user. The resumed session inherits settings from the original session.

## Following Up

- After every `codex` command, use `AskUserQuestion` to confirm next steps, collect clarifications, or decide whether to resume.
- When resuming, pipe the new prompt via stdin as shown above.
- Restate the chosen model and reasoning effort when proposing follow-up actions.

## Error Handling

- Stop and report failures whenever `codex --version` or a `codex exec` command exits non-zero; request direction before retrying.
- When output includes warnings or partial results, summarize them and ask how to adjust using `AskUserQuestion`.

## Example Commands

> **Remember**: All examples below require `dangerouslyDisableSandbox: true` in the Bash tool call!

**Coding task (refactoring)**:
```bash
# Bash tool: dangerouslyDisableSandbox: true
echo "Refactor the authentication module to use JWT tokens" | codex exec --skip-git-repo-check \
  -m gpt-5.2-codex \
  --config model_reasoning_effort=xhigh \
  --sandbox danger-full-access \
  --full-auto \
  2>/dev/null
```

**Analysis task (code review)**:
```bash
# Bash tool: dangerouslyDisableSandbox: true
echo "Review the error handling in src/routers/ and suggest improvements" | codex exec --skip-git-repo-check \
  -m gpt-5.2 \
  --config model_reasoning_effort=high \
  --sandbox danger-full-access \
  --full-auto \
  2>/dev/null
```

**Resume session**:
```bash
# Bash tool: dangerouslyDisableSandbox: true
echo "Now apply the suggested changes" | codex exec --skip-git-repo-check resume --last 2>/dev/null
```
