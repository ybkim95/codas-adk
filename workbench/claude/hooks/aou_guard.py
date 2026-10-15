#!/usr/bin/env python3
"""PreToolUse guard for Claude Code operating CoDaS inside an All of Us workspace.

Claude Code's role here is to operate CoDaS, not to look at data. CoDaS already keeps participant
rows away from its own agents; this hook keeps them away from Claude Code too, by blocking any tool
call that could put participant-level data into Claude Code's context:

* reading anything under .codas_runs/ except CoDaS's masked outputs (*_report.md, *_rounds.json,
  explore_summary.md);
* querying BigQuery or Cloud Storage directly, or running an interpreter on anything other than the
  CoDaS entry points (scripts/aou/run_codas_aou.py, the quickstart notebook, pytest, pip);
* recursive grep/find over the workspace (it would reach the participant tables);
* writing code that reads the tables, or editing the guard and the privacy module.

It is a best-effort allow-list, not a proof of isolation; the CLAUDE.md beside it tells Claude to
stop and ask rather than work around a block. Exit code 2 blocks the call and shows the reason.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

MASKED_OUTPUT = re.compile(r"(_report\.md|_rounds\.json|explore_summary\.md)$")
DATA_HINT = re.compile(r"codas_runs|\.csv\b|\.parquet\b")
DIRECT_ACCESS = re.compile(r"\bbq\b|bigquery|gsutil\s+(cat|cp)|gcloud\s+storage\s+(cat|cp)|INFORMATION_SCHEMA", re.I)
PROTECTED = re.compile(r"(^|[\s/'\"=])\.claude(/|\s|$)|codas/aou/privacy\.py|codas/agents/callbacks\.py")
CODE_SUFFIX = (".py", ".ipynb", ".sh", ".r", ".sql", ".js")
CODE_DATA_ACCESS = re.compile(r"bigquery|read_csv|read_parquet|codas_runs|\.csv\b|INFORMATION_SCHEMA|open\([^)]*\.csv", re.I)

INTERPRETERS = {"python", "python3", "ipython", "jupyter", "R", "Rscript", "node", "perl", "ruby", "sh",
                "bash", "zsh", "duckdb", "sqlite3", "papermill"}
ALLOWED_RUNS = [
    re.compile(r"^python3?\s+(\./)?scripts/aou/run_codas_aou\.py(\s|$)"),
    re.compile(r"^python3?\s+-m\s+(pytest|pip)(\s|$)"),
    re.compile(r"^jupyter\s+nbconvert\s+.*--execute\s+.*notebooks/aou_codas_quickstart\.ipynb"),
]
WRAPPERS = {"sudo", "time", "nohup", "exec", "env", "command", "nice"}
SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\||\n|\$\(|`|\(|\)")


def block(reason: str) -> None:
    sys.stderr.write(
        f"Blocked by the All of Us guard: {reason}\n"
        "Participant-level data must not enter this conversation. Use the CoDaS entry points "
        "(python scripts/aou/run_codas_aou.py ...) and read only the masked outputs. If you need "
        "something else, stop and ask the user instead of working around this."
    )
    sys.exit(2)


def _first_word(segment: str) -> tuple[str, str]:
    try:
        words = shlex.split(segment, posix=True)
    except ValueError:
        words = segment.split()
    while words and (words[0] in WRAPPERS or re.match(r"^\w+=", words[0])):
        words = words[1:]
    return (words[0] if words else ""), " ".join(words)


def check_bash(command: str) -> None:
    if PROTECTED.search(command):
        block("the command touches the guard configuration or the privacy module.")
    if DIRECT_ACCESS.search(command):
        block("direct BigQuery / Cloud Storage access prints rows into the conversation.")
    for segment in SEGMENT_SPLIT.split(command):
        segment = segment.strip()
        if not segment:
            continue
        word, normalized = _first_word(segment)
        base = word.rsplit("/", 1)[-1]
        if base in INTERPRETERS or base.startswith("python"):
            normalized = re.sub(r"^\S*/(python3?)\b", r"\1", normalized)
            if not any(p.search(normalized) for p in ALLOWED_RUNS):
                block(f"'{base}' may only run the CoDaS entry points, pytest or pip.")
            continue
        if base in {"grep", "egrep", "fgrep"} and re.search(r"(^|\s)-\w*[rR]\w*\b|--recursive", normalized) \
                and "--exclude-dir=.codas_runs" not in normalized:
            block("recursive grep would read the participant tables; add --exclude-dir=.codas_runs.")
        if base == "rg" and re.search(r"--no-ignore|(^|\s)-u+\b|--hidden", normalized):
            block("rg without .gitignore would read the participant tables.")
        if base == "find" and re.search(r"-exec(dir)?\b|-delete\b", normalized):
            block("find -exec / -delete is not allowed in this workspace.")
        segment = segment.replace("--exclude-dir=.codas_runs", "")  # excluding the tables is the safe form
        if DATA_HINT.search(segment):
            paths = [t for t in re.split(r"\s+", segment) if DATA_HINT.search(t)]
            if base in {"ls"} and not any(t.endswith((".csv", ".parquet")) and "*" not in t for t in paths[1:]):
                continue  # listing file names is fine
            if base in {"cat", "head", "tail", "less", "wc"} and paths and all(MASKED_OUTPUT.search(t) for t in paths):
                continue  # masked CoDaS outputs
            block("only CoDaS's masked outputs under .codas_runs/ may be read.")


def check_path_read(path: str) -> None:
    if "codas_runs" in path and not MASKED_OUTPUT.search(path):
        block("only CoDaS's masked outputs under .codas_runs/ may be read.")


def check_write(path: str, content: str) -> None:
    if PROTECTED.search(path):
        block("the guard configuration and the privacy module are read-only here.")
    if "codas_runs" in path:
        block("files under .codas_runs/ are written by CoDaS only.")
    if path.lower().endswith(CODE_SUFFIX) and CODE_DATA_ACCESS.search(content or ""):
        block("new code may not read the participant tables or query BigQuery; use the CoDaS entry points.")


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        block("could not parse the hook input.")
    tool = event.get("tool_name", "")
    args = event.get("tool_input", {}) or {}
    if tool == "Bash":
        check_bash(str(args.get("command", "")))
    elif tool == "Read":
        check_path_read(str(args.get("file_path", "")))
    elif tool == "Grep":
        check_path_read(str(args.get("path", "") or ""))
        if "codas_runs" in str(args.get("glob", "") or ""):
            block("only CoDaS's masked outputs under .codas_runs/ may be read.")
    elif tool in {"Write", "Edit", "NotebookEdit"}:
        path = str(args.get("file_path") or args.get("notebook_path") or "")
        content = " ".join(str(args.get(k, "") or "") for k in ("content", "new_string", "new_source"))
        check_write(path, content)
    sys.exit(0)


if __name__ == "__main__":
    main()
