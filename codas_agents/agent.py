"""Google ADK agent definition for CoDaS.

The pipeline is intentionally tool-grounded: every statistic is computed by the deterministic
Python runners in ``codas_core``. The LLM may plan, profile, choose the target/roles, explain, and
ask for human input, but it must not invent reportable numbers. Agent instructions live in
``prompts.py``; session/execution is in ``runtime.py``; guardrails and logging are in
``callbacks.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from google.adk.agents import LlmAgent, SequentialAgent

from codas_agents import prompts
from codas_agents.callbacks import after_tool_logger, before_model_logger, before_tool_guardrail
from codas_core.settings import load_local_env
from codas_core.data import profile_dataframe, read_csv_dataset
from codas_core.discovery import DiscoveryRequest, run_discovery_from_csv

load_local_env()

MODEL = os.getenv("CODAS_GEMINI_MODEL", "gemini-3.5-flash")


# --- Deterministic tools (the only source of numbers) ------------------------------------------

def profile_dataset(csv_path: str, target_column: str | None = None) -> dict[str, Any]:
    """Profile a CSV: schema, dtypes, missingness, and the numeric columns (candidate targets)."""
    df = read_csv_dataset(Path(csv_path).expanduser().resolve())
    return profile_dataframe(df, target_column=target_column).to_dict()


def run_discovery(
    csv_path: str,
    target_column: str,
    participant_id_column: str | None = None,
    time_column: str | None = None,
    excluded_columns_csv: str = "",
    confounder_columns_csv: str = "",
    top_k: int = 15,
) -> dict[str, Any]:
    """Run deterministic association discovery for an explicit target column.

    Roles are taken as given: the engine infers nothing from column names. Pass comma-separated
    column names for excluded/confounder columns; leave participant/time empty if not applicable.
    """
    excluded = [v.strip() for v in excluded_columns_csv.split(",") if v.strip()]
    confounders = [v.strip() for v in confounder_columns_csv.split(",") if v.strip()]
    request = DiscoveryRequest(
        target_column=target_column,
        participant_id_column=participant_id_column,
        time_column=time_column,
        excluded_columns=excluded,
        confounder_columns=confounders,
        top_k=top_k,
        validation_resamples=int(os.getenv("CODAS_VALIDATION_RESAMPLES", "1000")),
    )
    return run_discovery_from_csv(Path(csv_path).expanduser().resolve(), request).to_dict()


def preview_columns(csv_path: str, limit: int = 20) -> dict[str, Any]:
    """Return a small preview of column names, dtypes, and the first few rows."""
    df = pd.read_csv(Path(csv_path).expanduser().resolve(), nrows=max(1, min(limit, 200)))
    return {
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "preview_rows": df.head(3).to_dict(orient="records"),
    }


# --- Agents ------------------------------------------------------------------------------------

def _agent(name: str, description: str, instruction: str, tools: list | None = None) -> LlmAgent:
    """Construct an LlmAgent with the shared logging/guardrail callbacks."""
    kwargs: dict[str, Any] = dict(
        name=name,
        model=MODEL,
        description=description,
        instruction=instruction,
        before_model_callback=before_model_logger,
    )
    if tools:
        kwargs.update(
            tools=tools,
            before_tool_callback=before_tool_guardrail,
            after_tool_callback=after_tool_logger,
        )
    return LlmAgent(**kwargs)


scout_agent = _agent(
    "scout_agent",
    "Profiles the dataset and identifies the target and the participant/time/confounder roles.",
    prompts.SCOUT,
    tools=[profile_dataset, preview_columns],
)
data_profiler_agent = _agent(
    "data_profiler_agent",
    "Audits schema, missingness, dtypes, row counts, and candidate targets before modeling.",
    prompts.DATA_PROFILER,
    tools=[profile_dataset, preview_columns],
)
empirical_agent = _agent(
    "stat_ml_agent",
    "Runs the deterministic statistical/ML discovery tool.",
    prompts.EMPIRICAL,
    tools=[run_discovery],
)
validation_agent = _agent(
    "validation_agent",
    "Reviews the deterministic validation battery (replication, stability, confounding, construct).",
    prompts.VALIDATION,
)
defender_agent = _agent(
    "defender_agent",
    "Defends candidates only when deterministic evidence supports retention.",
    prompts.DEFENDER,
)
review_agent = _agent(
    "critic_reviewer_agent",
    "Critiques leakage, confounding, reporting integrity, and interpretation boundaries.",
    prompts.REVIEW,
)
mechanism_agent = _agent(
    "mechanism_agent",
    "Links surviving candidates to plausible mechanisms and interpretation boundaries.",
    prompts.MECHANISM,
)
novelty_agent = _agent(
    "novelty_agent",
    "Assesses whether candidate operationalizations are established, supported, emerging, or unverified.",
    prompts.NOVELTY,
)
strategy_agent = _agent(
    "strategy_agent",
    "Recommends accept / reinvestigate / deeper-analysis / human-feedback from the run evidence.",
    prompts.STRATEGY,
)
artifact_agent = _agent(
    "artifact_agent",
    "Assembles tables, figures, fact sheets, and audit traces from grounded outputs.",
    prompts.ARTIFACT,
)
report_agent = _agent(
    "report_agent",
    "Drafts a publication-style summary from the deterministic Fact Sheet and artifacts.",
    prompts.REPORT,
)


root_agent = SequentialAgent(
    name="codas_orchestrator",
    description="CoDaS orchestrator for deterministic, leakage-guarded association discovery.",
    sub_agents=[
        scout_agent,
        data_profiler_agent,
        empirical_agent,
        validation_agent,
        review_agent,
        defender_agent,
        mechanism_agent,
        novelty_agent,
        strategy_agent,
        artifact_agent,
        report_agent,
    ],
)
