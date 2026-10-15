#!/usr/bin/env bash
# Install the All of Us guard for Claude Code into this checkout's .claude/ (git-ignored).
#
# Claude Code is pre-installed in Workbench cloud apps. With this guard it runs on Vertex AI in the
# workspace project and can operate CoDaS without reading participant-level data. Whether All of Us
# accepts an AI coding agent in a workspace at all is not settled: install this only after the
# program has confirmed it in writing, and pass --confirmed to say so.
set -euo pipefail

if [[ "${1:-}" != "--confirmed" ]]; then
  cat >&2 <<'EOF'
Not installed. All of Us has not published guidance on AI coding agents in workspaces, and the
Data User Code of Conduct forbids giving workspace access to any AI tool that could export
participant-level data. Get written confirmation from the program (support@researchallofus.org)
first, then run:

    scripts/aou/install_claude_guard.sh --confirmed
EOF
  exit 1
fi

repo="$(cd "$(dirname "$0")/../.." && pwd)"
project="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
if [[ -z "$project" ]]; then
  echo "No Google Cloud project found; set GOOGLE_CLOUD_PROJECT to the workspace project." >&2
  exit 1
fi

mkdir -p "$repo/.claude/hooks"
cp "$repo/workbench/claude/hooks/aou_guard.py" "$repo/.claude/hooks/aou_guard.py"
cp "$repo/workbench/claude/CLAUDE.md" "$repo/.claude/CLAUDE.md"
sed "s/__WORKSPACE_PROJECT__/$project/g" "$repo/workbench/claude/settings.json" > "$repo/.claude/settings.json"
chmod +x "$repo/.claude/hooks/aou_guard.py"

for key in ANTHROPIC_API_KEY GOOGLE_API_KEY GEMINI_API_KEY; do
  if [[ -n "${!key:-}" ]]; then
    echo "Warning: $key is set. Unset it so Claude Code and CoDaS use Vertex AI only." >&2
  fi
done
echo "Installed into $repo/.claude (Vertex project: $project). Start Claude Code from $repo and check /status shows Vertex AI."
