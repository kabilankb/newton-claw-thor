# sandbox/ — egress policy for sandboxed runtimes

`policies/newton-control-policy.yaml` lets a NemoClaw/OpenShell sandbox reach the host
**Newton Control Server** at `host.openshell.internal:5561` (the only port this system uses
besides the viser viewer on `8090`).

```bash
nemoclaw <sandbox> policy-add newton-control --from-file sandbox/policies/newton-control-policy.yaml --yes
```

`agents/nemoclaw/install.sh <sandbox>` applies it together with the `newton__*` MCP tools —
see `agents/nemoclaw/RUNBOOK.md`.
