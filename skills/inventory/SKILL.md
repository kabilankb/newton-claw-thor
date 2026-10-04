---
name: "newton-inventory"
description: "Answer 'what robots / tasks exist?' from the authoritative inventory — never from memory. Use for 'list robots', 'what can I train', 'how many tasks', 'do you have a Spot?', 'is there a pretrained policy for X'. Trigger keywords: list, what robots, which robots, what tasks, available, inventory, how many, do you have, is there, catalog, supported."
user-invocable: true
---

# Inventory — RUN ONE COMMAND, answer only from its output

```bash
~/newton-claw/agents/openclaw/claw tasks      # trainable tasks (add a word to filter: claw tasks go2)
~/newton-claw/agents/openclaw/claw robots     # robots the viewer can open
~/newton-claw/agents/openclaw/claw runs       # policies already trained on this Thor
```
The same lists are in `INVENTORY.md` next to this file (generated from the real registries).

## Rules
- Answer ONLY with names that appear in the command output.
- If the user asks for something not listed (e.g. Spot, a warehouse scene), say it is
  not available on this Newton system. Never substitute another robot silently.
