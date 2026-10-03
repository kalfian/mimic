#!/usr/bin/env python3
"""Fake ``claude`` CLI for interpreter tests (point ``MIMIC_CLAUDE_BIN`` at a wrapper of it).

Mimics the Claude Code 2.1.287 ``-p --output-format json --json-schema`` envelope (see
``app/interpret/claude_cli.py`` docstring). Stdlib only.

Environment:

* ``FAKE_CLAUDE_MODE``:
  ``ok`` (default) — valid envelope with ``structured_output`` built from the schema's id enum;
  ``envelope_result_only`` — no ``structured_output``, payload JSON string in ``result``;
  ``braces`` — ``result`` is prose with the payload embedded (balanced-brace fallback);
  ``bad_ids`` — unknown ids, bad role, invalid parent/target, control chars, over-long label;
  ``garbage`` — malformed stdout (not JSON);
  ``is_error`` — exit 0 but ``is_error: true``;
  ``error`` — exit 1 with a stderr message;
  ``timeout`` — spawns a grandchild in the same process group, then sleeps forever-ish.
* ``FAKE_CLAUDE_RECORD``: write ``{"argv": [...], "cwd": "..."}`` JSON to this path.
* ``FAKE_CLAUDE_PIDFILE``: (timeout mode) write ``"<child_pid> <grandchild_pid>"`` here.
* ``FAKE_CLAUDE_SLEEP``: seconds to sleep before answering (non-timeout modes).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


def _arg(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        i = argv.index(flag) + 1
        if i < len(argv):
            return argv[i]
    return None


def _ids_from_schema(schema: dict) -> list[str]:
    try:
        return list(schema["properties"]["elements"]["items"]["properties"]["id"]["enum"])
    except (KeyError, TypeError):
        return []


def _good_payload(ids: list[str]) -> dict:
    roles = ["card", "image", "title", "button", "text", "icon", "badge", "other"]
    elements = []
    for i, el_id in enumerate(ids):
        elements.append(
            {
                "id": el_id,
                "label": f"Fake {roles[i % len(roles)]} {el_id}",
                "role": roles[i % len(roles)],
                "parent_id": ids[0] if i > 0 else None,
                "description": f"Fake description for {el_id}",
                "confidence": 0.9,
            }
        )
    return {
        "elements": elements,
        "interaction": {
            "type": "hover",
            "type_confidence": 0.85,
            "target_element_id": ids[0] if ids else None,
            "target_description": "A product card",
            "trigger_description": "Pointer enters the card",
        },
        "structure": ["image", "title", "price"],
        "notes": ["fake note"],
    }


def _bad_payload(ids: list[str]) -> dict:
    p = _good_payload(ids)
    first = p["elements"][0]
    first["label"] = "Bad\x00label\x1b[31m " + "x" * 120
    first["role"] = "spaceship"
    first["parent_id"] = "e77"
    p["elements"].append(
        {"id": "e99", "label": "Ghost", "role": "card", "parent_id": None, "confidence": 0.5}
    )
    p["interaction"]["target_element_id"] = "e42"
    p["interaction"]["type"] = "teleport"
    p["structure"] = ["image", "IMAGE", "", "y" * 80]
    return p


def _envelope(**over: object) -> dict:
    env = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 2,
        "duration_ms": 1234,
        "stop_reason": "tool_use",
        "terminal_reason": "completed",
        "permission_denials": [],
        "session_id": "00000000-0000-0000-0000-000000000000",
        "modelUsage": {"claude-fake-sonnet": {"inputTokens": 1, "outputTokens": 1}},
        "total_cost_usd": 0.0,
    }
    env.update(over)
    return env


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--version"]:
        print("2.1.287 (Claude Code fake)")
        return 0

    record = os.environ.get("FAKE_CLAUDE_RECORD")
    if record:
        with open(record, "w", encoding="utf-8") as f:
            json.dump({"argv": argv, "cwd": os.getcwd()}, f)

    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    if mode == "timeout":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        pidfile = os.environ.get("FAKE_CLAUDE_PIDFILE")
        if pidfile:
            with open(pidfile, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()} {child.pid}")
        time.sleep(120)
        return 0

    time.sleep(float(os.environ.get("FAKE_CLAUDE_SLEEP", "0")))
    schema = json.loads(_arg(argv, "--json-schema") or "{}")
    ids = _ids_from_schema(schema)

    if mode == "garbage":
        sys.stdout.write("Thinking...\n{not json at all\n")
        return 0
    if mode == "error":
        sys.stderr.write("Error: fake failure (not logged in)\n")
        return 1
    if mode == "is_error":
        print(json.dumps(_envelope(subtype="error_during_execution", is_error=True, result="")))
        return 0

    if "shape" in schema.get("properties", {}):  # check_interpreter(deep=True) probe
        payload: dict = {"shape": "rectangle", "color": "blue"}
    else:
        payload = _bad_payload(ids) if mode == "bad_ids" else _good_payload(ids)
    text = json.dumps(payload)
    if mode == "envelope_result_only":
        env = _envelope(result=text)
    elif mode == "braces":
        env = _envelope(result=f"Here are the labels you asked for:\n{text}\nHope this helps.")
    else:  # ok / bad_ids
        env = _envelope(result=text, structured_output=payload)
    print(json.dumps(env))
    return 0


if __name__ == "__main__":
    sys.exit(main())
