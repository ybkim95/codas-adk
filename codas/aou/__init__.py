"""All of Us Researcher Workbench integration: cohort extraction and the AoU privacy guard.

Everything in this package is meant to run INSIDE the Researcher Workbench. The participant-level
analysis table it builds stays on the workspace disk; only aggregate, small-cell-masked values ever
reach a language model or a report (see ``privacy``).
"""
