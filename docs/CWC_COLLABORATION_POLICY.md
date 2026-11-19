# CWC Collaboration Policy

Revision R1, authorized 2026-11-19 UTC by ISRJ2_CWC_AUTONOMOUS_SUPERVISOR_R1. This supersedes earlier CWC suspension clauses. Other experiment constraints continue to apply.

## Scope and identity

Use CWC for complex software diagnosis after two evidenced attempts, architecture choices, evidence contradictions, changes to state or recovery logic, and independent review. Routine hashes and log checks need no consultation. Codex owns execution and validates advice against current authorization. ChatGPT accesses the workspace through the read-only connector.

The workspace is happy-liskov, local branch cindy/happy-liskov, connector Codex with ChatGPT · happy-liskov. All conversations belong to the codex-with-chatgpt Project. The Project also has other workspace conversations, so its general instructions must require the connector explicitly named by each task. Never substitute serene-bohr. On a new chat, require workspace_info to identify happy-liskov before accepting a plan.

Use the installed c2c CLI, version 0.1.3 at this revision. Consult command help before using unfamiliar options. The upstream protocol has no numbered protocol release; record the CLI version and upstream file blob instead of inventing a protocol version. Skill blob inspected on 2026-11-19 is 0fd0bc863836a6af75ebf2688106fe16eb2e5857.

## Protocol and checkpoints

1. Read AGENTS.md, the current handoff, collaboration log, and c2c session. Resume an existing checkpoint before assigning a task ID.
2. Run c2c doctor for this workspace. Preserve a healthy connection. Confirm the built-in browser, existing Project chat, and exact connector.
3. Send a control message under 1 KB, beginning [C2C], STATE: INIT, TASK_ID and ITERATION: 0. Include a bounded goal and file references. Files, diffs and logs are read through the connector.
4. Confirm the message is visible, then save the official INIT/GPT_PLAN checkpoint. Record dispatch intent before sending and confirmed visibility afterward. Unknown delivery requires inspection of the same chat, never a blind resend.
5. Accept PLAN only when Task ID and iteration match. Record the actual received UTC, workspace verification, rationale, allowed actions and success criteria.
6. Codex executes allowed checks, saves test output locally, and uses c2c record with command, output file and exit status. Record failed checks too.
7. Save EXECUTED_LOCAL, send EXECUTED with the same task and iteration, confirm visibility and save EXECUTED_SENT/GPT_REVIEW. Request independent evidence inspection. REVIEW may be implicit in the official protocol.
8. Verify DONE against actual evidence and task identity. Preserve the reply and decision summary, then clear the official checkpoint. A bare DONE cannot establish missing acceptance evidence.
9. BLOCKED or ERROR preserves the reason and completed work. Continue independent management. Default maximum is 12 iterations per task unless the saved configuration explicitly differs. At the limit, retain the task ID and seek a decision.

## Recovery and unattended operation

For connection failure, record a sanitized error, run official Doctor and inspect session identity. Make at most two evidence-based recovery attempts for the same fault. Login, CAPTCHA, 2FA and account consent require the user. Record CWC_UNAVAILABLE if recovery fails. Never print credentials, cookies or pairing material.

For INIT or EXECUTED_SENT, inspect the saved chat and await the corresponding reply. For EXECUTED_LOCAL, send only the pending result. For EXECUTING, continue its saved plan. A lost chat is recovered with HANDOFF within the same Project. Preserve the previous URL until the replacement confirms identity. Do not re-pair or replace a healthy connector just to resume.

Keep the Cindy task bound to its current provider and model route. Inspect runtime generation and pending changes; leave identity validation enabled. A provider mismatch requires supported session recovery, never bypassing the check. Save its failure receipt and continue work through available independent mechanisms.

## Natural-trigger acceptance

Task c2c_a719 uses only outputs/status_audit/collab_fixture/temperature.py. A one-time Cindy timer injects it into the current session. This is separate from the sole recurring hourly supervisor. Verify the timer's actual run receipt before sending INIT. Record natural trigger, dispatch, PLAN receipt, independent unit checks, EXECUTED, DONE and a subsequent management inspection. Record all UTC times and task/iteration identities.

The previous review remains MANUAL_READ_ONLY_REVIEW_VERIFIED. Set AUTO_CWC_ENABLED only after this whole chain has evidence, including management continuation and absence of duplicate sends. One successful acceptance does not prove indefinite service availability.

## Hourly ownership and persistence

Keep one recurring supervisor at 0 * * * * UTC. Its agent reads these rules, invokes the existing hourly audit, records actual trigger and provider runtime, checks CWC health, and addresses one bounded independent management item. CWC failure is recorded while local audit and permitted reporting continue. Use the existing hour lock and slot ledger. Repeat invocations must inspect the slot before publication.

Report startup count, verified fixed milestones, and verified training-plus-evaluation completion separately. Preserve every historic snapshot. Server failures have their own result and time. An independent server query cannot change an earlier hour's failure. Evidence conflicts or unavailable services go to the pending-decision register.

Use reports/CWC_COLLABORATION_LOG.md for task, trigger, advice, adopted action, validation, outcome and next step. Local sanitized receipts under outputs/status_audit remain recoverable evidence; public reports contain summaries only. Update handoff/CURRENT_STATE.md at material changes. Save durable user preferences with Cindy memory. Read current files again after interruption.

## Publication and confidentiality

Only paper2 and codex/1103-experiment are authorized. Inspect remote head before publication. Use exact paths and a separate index or document checkout, preserve the full remote AGENTS and handoff, and verify the pushed SHA. Local short AGENTS must never replace the remote full rule set.

Keep credentials, raw datasets, arrays, weights, checkpoints and full logs out of GitHub and control messages. Do not open old held-out Gate arrays. CWC advice cannot expand GPU budgets, authorize deployment, change frozen protocols or permit training restarts.

Official references are [skill](https://github.com/XiaoDuoYa/codex-with-chatgpt/blob/main/skill/SKILL.md) and [protocol](https://github.com/XiaoDuoYa/codex-with-chatgpt/blob/main/docs/protocol.md).
