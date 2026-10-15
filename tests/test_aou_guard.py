"""The Claude Code PreToolUse guard for All of Us workspaces, exercised as Claude Code calls it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "workbench" / "claude" / "hooks" / "aou_guard.py"


def _run(tool: str, **tool_input: str) -> subprocess.CompletedProcess:
    payload = json.dumps({"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input})
    return subprocess.run([sys.executable, str(HOOK)], input=payload, capture_output=True, text=True, timeout=30)


ALLOWED_BASH = [
    "python scripts/aou/run_codas_aou.py explore --screen-only",
    "python3 scripts/aou/run_codas_aou.py run --csv .codas_runs/aou/explore_hba1c.csv --goal 'HbA1c (lab)'",
    "/opt/conda/bin/python scripts/aou/run_codas_aou.py explore --top-k 5",
    "python -m pytest -q tests/test_aou.py",
    "pip install -e '.[aou]'",
    "ls -la .codas_runs/aou",
    "cat .codas_runs/aou/explore_summary.md",
    "head -50 .codas_runs/aou/explore_hba1c_report.md",
    "git status && git diff codas/aou/cohort.py",
    "rg 'def explore' codas",
    "grep -rn --exclude-dir=.codas_runs 'feasibility' .",
    "jupyter nbconvert --to notebook --execute notebooks/aou_codas_quickstart.ipynb",
]

BLOCKED_BASH = [
    "head .codas_runs/aou/explore_hba1c.csv",
    "cat .codas_runs/aou/*.csv | wc -l",
    "python -c \"import pandas as pd; print(pd.read_csv('.codas_runs/aou/x.csv').head())\"",
    "python analysis.py",
    "bq query --use_legacy_sql=false 'SELECT * FROM `cdr.person` LIMIT 5'",
    "bq head fc-aou-cdr-prod:R2025Q4R6.measurement",
    "gsutil cat gs://bucket/data.csv",
    "grep -r 5.7 .",
    "rg --no-ignore hba1c",
    "find . -name '*.csv' -exec head {} \\;",
    "duckdb -c \"select * from '.codas_runs/aou/x.csv'\"",
    "ls && python3 - <<'EOF'\nprint(1)\nEOF",
    "rm .claude/settings.json",
    "sed -i 's/aou_mode()/False/' codas/aou/privacy.py",
    "awk 'NR<5' .codas_runs/aou/explore_t2d.csv",
    "echo $(cat .codas_runs/aou/explore_t2d.csv)",
]


@pytest.mark.parametrize("command", ALLOWED_BASH)
def test_bash_allowed(command):
    result = _run("Bash", command=command)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", BLOCKED_BASH)
def test_bash_blocked(command):
    result = _run("Bash", command=command)
    assert result.returncode == 2 and "All of Us guard" in result.stderr


@pytest.mark.parametrize("path,code", [
    (".codas_runs/aou/explore_hba1c.csv", 2),
    ("/home/jupyter/codas-adk/.codas_runs/aou/hba1c_fitbit.csv", 2),
    (".codas_runs/aou/explore_hba1c_report.md", 0),
    (".codas_runs/aou/explore_hba1c_rounds.json", 0),
    ("codas/aou/cohort.py", 0),
])
def test_read(path, code):
    assert _run("Read", file_path=path).returncode == code


def test_grep_into_participant_tables_blocked():
    assert _run("Grep", pattern="hba1c", path=".codas_runs/aou").returncode == 2
    assert _run("Grep", pattern="hba1c", glob=".codas_runs/**/*.csv").returncode == 2
    assert _run("Grep", pattern="def explore", path="codas").returncode == 0


@pytest.mark.parametrize("tool,args,code", [
    ("Write", {"file_path": "analysis.py", "content": "import pandas as pd\npd.read_csv('.codas_runs/aou/x.csv')"}, 2),
    ("Write", {"file_path": "q.py", "content": "from google.cloud import bigquery"}, 2),
    ("Write", {"file_path": ".codas_runs/aou/notes.md", "content": "x"}, 2),
    ("Edit", {"file_path": "codas/aou/privacy.py", "old_string": "a", "new_string": "b"}, 2),
    ("Edit", {"file_path": ".claude/settings.json", "old_string": "a", "new_string": "b"}, 2),
    ("Edit", {"file_path": "scripts/aou/outcomes_catalog.json", "old_string": "a", "new_string": "b"}, 0),
    ("Write", {"file_path": "docs/notes.md", "content": "the run read explore_hba1c.csv"}, 0),
    ("NotebookEdit", {"notebook_path": "notebooks/x.ipynb", "new_source": "pd.read_csv('a.csv')"}, 2),
])
def test_writes(tool, args, code):
    assert _run(tool, **args).returncode == code


def test_settings_template_is_valid_and_wires_the_hook():
    settings = json.loads((ROOT / "workbench" / "claude" / "settings.json").read_text())
    assert settings["env"]["CLAUDE_CODE_USE_VERTEX"] == "1"
    command = settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    assert ".claude/hooks/aou_guard.py" in command


def test_installer_refuses_without_confirmation():
    result = subprocess.run(["bash", str(ROOT / "scripts" / "aou" / "install_claude_guard.sh")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1 and "written confirmation" in result.stderr
