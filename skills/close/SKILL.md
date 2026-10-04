---
name: "newton-close"
description: "Stop whatever is running on the Jetson Thor GPU — Newton training, inference/play, evaluation, or the viewer — and free the GPU. Trigger keywords: close, stop, shut down, shutdown, kill, end, quit, stop training, stop the sim, stop inference, close the viewer, free gpu."
user-invocable: true
---

# Close / Stop — RUN ONE COMMAND

⚠️ EXECUTE skill. No plan, no prose first. **Run this with your shell/exec tool**:
```bash
~/newton-claw/agents/openclaw/claw close
```
It stops the one running job (training OR inference OR viewer) and frees the GPU.
Safe anytime; if nothing is running it says so.

## After it runs
Report the result (`closed — GPU free`). To check first: `~/newton-claw/agents/openclaw/claw status`.
