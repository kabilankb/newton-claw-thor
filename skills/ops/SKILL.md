---
name: "newton-claw-ops"
description: "Bootstrap and health-check newton-claw on the Jetson Thor. Use FIRST when a control call fails to connect, or for 'is it running', 'restart the server', 'check the Thor', 'how much memory/disk is free'. Trigger keywords: start, bring up, boot, is it running, server down, connection refused, health, status, restart, cant connect, control server, device, memory, disk, jetson, thor."
user-invocable: true
---

# newton-claw Ops — RUN ONE COMMAND

```bash
~/newton-claw/agents/openclaw/claw up        # start the control server if it is down (idempotent)
~/newton-claw/agents/openclaw/claw status    # what is running on the GPU right now
~/newton-claw/agents/openclaw/claw device    # Thor model, free memory, free disk
~/newton-claw/agents/openclaw/claw restart   # restart the control server (after code/config changes)
```

## Self-heal rule
If ANY `claw` command reports "connection refused" or no response: run `claw up`, then
retry the original command once.

## Facts
- One control server on `http://localhost:5561`; it launches every GPU job.
- ONE GPU job at a time (training OR inference OR viewer). A second returns 409 → `claw close` first.
- Thor memory is unified (CPU + GPU + this language model share 128 GB). If memory is
  low, lower `--num_envs` (4096 → 2048 → 1024).
- For deeper Jetson diagnostics (thermals, power mode, memory audit) use the
  `jetson-diagnostic` and `jetson-memory-audit` skills.
