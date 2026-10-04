#!/usr/bin/env python3
"""Talk to a local model (Gemma via Ollama) and let it train / run robots in Newton.

A minimal terminal agent using the SAME `newton__*` tools as the NemoClaw sandbox
(agents/nemoclaw/newton-mcp.mjs), for when the sandbox is not available. Stdlib only.

    agents/chat.py                       # interactive
    agents/chat.py "train the Go2"       # one-shot

Env: NEWTON_LLM_URL (default http://localhost:8000), NEWTON_LLM_MODEL (default gemma4:12b).
"""
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MCP = os.path.join(HERE, "nemoclaw", "newton-mcp.mjs")
GUIDE = open(os.path.join(HERE, "nemoclaw", "AGENTS-newton.md")).read()
URL = os.environ.get("NEWTON_LLM_URL", "http://localhost:8000")
MODEL = os.environ.get("NEWTON_LLM_MODEL", "gemma4:12b")
ENV = {**os.environ, "NEWTON_CONTROL_HOST": os.environ.get("NEWTON_CONTROL_HOST", "localhost")}


def mcp(method, params=None):
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}) + "\n"
    out = subprocess.run(["node", "--no-warnings", MCP], input=req, capture_output=True, text=True,
                         timeout=60, env=ENV).stdout
    return json.loads(out.splitlines()[0])["result"]


def llm(messages, tools):
    body = {"model": MODEL, "stream": False, "options": {"temperature": 0, "num_ctx": 16384},
            "messages": messages, "tools": tools}
    req = urllib.request.Request(URL + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]


def turn(messages, tools, text):
    messages.append({"role": "user", "content": text})
    for _ in range(4):                                   # at most a few tool hops per request
        msg = llm(messages, tools)
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            return msg.get("content", "").strip()
        for c in calls:
            name, args = c["function"]["name"], c["function"].get("arguments") or {}
            print(f"  [tool] {name} {json.dumps(args)}")
            result = mcp("tools/call", {"name": name.replace("newton__", "", 1), "arguments": args})
            messages.append({"role": "tool", "tool_name": name, "content": result["content"][0]["text"][:6000]})
    return "(stopped after several tool calls)"


def main():
    subprocess.run([os.path.join(HERE, "openclaw", "claw"), "up"], stdout=subprocess.DEVNULL)
    tools = [{"type": "function", "function": {"name": "newton__" + t["name"], "description": t["description"],
              "parameters": t["inputSchema"]}} for t in mcp("tools/list")["tools"]]
    messages = [{"role": "system", "content": GUIDE}]
    if len(sys.argv) > 1:
        print(turn(messages, tools, " ".join(sys.argv[1:])))
        return
    print(f"newton-claw chat — {MODEL} @ {URL}. Try: train the Go2 · how is it going? · stop it   (Ctrl-D to quit)")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text:
            print("agent> " + turn(messages, tools, text))


if __name__ == "__main__":
    main()
