"""Compile one run's outputs into a PDF: the final report, every phase's output, the per-round debate,
and an appendix with the fact sheet, phase coverage and the candidates' effect sizes.

Pure Python (``fpdf2`` + ``markdown``), so it needs no LaTeX or system libraries in the Workbench. A
Unicode TrueType font is used when one is installed; otherwise text is transliterated to Latin-1.
Everything it prints comes from files the run already wrote (masked for counts of 1-20).
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path
from typing import Any

# Font files tried in order: (regular, bold, italic, bold-italic, monospace). The first family whose
# regular face exists is used; missing styles fall back to regular.
_FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
              "/usr/local/share/fonts", "/Library/Fonts", str(Path.home() / ".fonts"))
_FAMILIES = (("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf",
              "DejaVuSansMono.ttf"),
             ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf", "NotoSans-BoldItalic.ttf",
              "NotoSansMono-Regular.ttf"),
             ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf",
              "LiberationSans-BoldItalic.ttf", "LiberationMono-Regular.ttf"))
# LaTeX-style math that LLM-written reports use; rendered as plain symbols.
_TEX = {r"\rho": "ρ", r"\alpha": "α", r"\beta": "β", r"\Delta": "Δ", r"\approx": "≈", r"\times": "×",
        r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\pm": "±", r"\%": "%", r"\rightarrow": "→",
        r"\to": "→", r"\sim": "~", r"\mu": "μ", r"\sigma": "σ", r"\chi": "χ"}
_LATIN1 = {"ρ": "rho", "α": "alpha", "β": "beta", "Δ": "Delta", "≈": "~", "×": "x", "≤": "<=", "≥": ">=",
           "±": "+/-", "→": "->", "←": "<-", "μ": "mu", "σ": "sigma", "χ": "chi", "—": "-", "–": "-", "‘": "'",
           "’": "'", "“": '"', "”": '"', "…": "...", "•": "-", "²": "^2", "√": "sqrt", "≠": "!="}


def _find_fonts() -> list[Path] | None:
    for family in _FAMILIES:
        for d in _FONT_DIRS:
            regular = Path(d) / family[0]
            if regular.exists():
                return [Path(d) / f if (Path(d) / f).exists() else regular for f in family]
    return None


def _clean_tex(text: str) -> str:
    for k in sorted(_TEX, key=len, reverse=True):
        text = text.replace(k, _TEX[k])
    text = re.sub(r"\^\{?2\}?", "²", text)
    return re.sub(r"\$([^$\n]{1,80})\$", r"\1", text)


class _Doc:
    def __init__(self, title: str):
        from fpdf import FPDF

        self.pdf = FPDF(format="A4")
        self.pdf.set_auto_page_break(True, margin=15)
        self.pdf.set_title(title)
        fonts = _find_fonts()
        if fonts:
            for style, path in zip(("", "B", "I", "BI"), fonts[:4]):
                self.pdf.add_font("Body", style, str(path))
            self.pdf.add_font("Mono", "", str(fonts[4]))
            self.family, self.mono, self.unicode = "Body", "Mono", True
        else:
            self.family, self.mono, self.unicode = "helvetica", "courier", False

    def text(self, value: str) -> str:
        value = _clean_tex(str(value))
        if self.unicode:
            return value
        for k, v in _LATIN1.items():
            value = value.replace(k, v)
        return value.encode("latin-1", "replace").decode("latin-1")

    def heading(self, value: str, level: int = 1) -> None:
        self.pdf.set_font(self.family, "B", {1: 16, 2: 13, 3: 11}.get(level, 11))
        self.pdf.multi_cell(0, 8 if level == 1 else 7, self.text(value), new_x="LMARGIN", new_y="NEXT")
        self.pdf.ln(1)

    def markdown(self, value: str) -> None:
        """Render LLM-written Markdown; fall back to plain text if the HTML renderer rejects it."""
        import markdown as md

        value = self.text(value or "").strip()
        if not value:
            self.plain("(no output)")
            return
        html = md.markdown(value, extensions=["tables", "sane_lists"])
        try:
            self.pdf.set_font(self.family, "", 10)
            self.pdf.write_html(html, font_family=self.family, pre_code_font=self.mono,
                                warn_on_tags_not_matching=False)
        except Exception:  # noqa: BLE001 - a rendering quirk must not lose the content
            self.plain(re.sub(r"<[^>]+>", "", html))
        self.pdf.ln(2)

    def plain(self, value: str, size: int = 10, style: str = "") -> None:
        self.pdf.set_font(self.family, style, size)
        self.pdf.multi_cell(0, 5, self.text(value), new_x="LMARGIN", new_y="NEXT")

    def table(self, header: list[str] | None, rows: list[list[Any]], widths: list[float] | None = None) -> None:
        self.pdf.set_font(self.family, "", 8)
        with self.pdf.table(col_widths=widths, text_align="LEFT", line_height=4.5,
                            first_row_as_headings=header is not None) as t:
            if header is not None:
                r = t.row()
                for h in header:
                    r.cell(self.text(h))
            for row in rows:
                r = t.row()
                for cell in row:
                    r.cell(self.text("" if cell is None else str(cell)))
        self.pdf.ln(3)

    def bars(self, items: list[tuple[str, float]]) -> None:
        """Horizontal bar chart of effect sizes, centred at zero (negative to the left)."""
        if not items:
            self.plain("(no candidates)")
            return
        pdf = self.pdf
        label_w, chart_w, h = 95.0, 80.0, 5.0
        top = max(abs(v) for _, v in items) or 1.0
        pdf.set_font(self.family, "", 7)
        x0 = pdf.l_margin + label_w + chart_w / 2
        for label, value in items:
            if pdf.get_y() > pdf.h - 25:
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y)
            pdf.cell(label_w - 2, h, self.text(label[:60]), align="R")
            width = chart_w / 2 * abs(value) / top
            pdf.set_fill_color(70, 110, 170) if value >= 0 else pdf.set_fill_color(190, 90, 60)
            pdf.rect(x0 if value >= 0 else x0 - width, y + 0.8, width, h - 1.6, style="F")
            pdf.set_xy(pdf.l_margin + label_w + chart_w + 1, y)
            pdf.cell(20, h, f"{value:+.3f}")
            pdf.ln(h)
        pdf.set_draw_color(120, 120, 120)
        pdf.line(x0, pdf.get_y() - h * len(items), x0, pdf.get_y())
        pdf.set_draw_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)  # later tables must not inherit the bar colour
        pdf.ln(3)


    def forest(self, items: list[tuple[str, float, float, float, bool]], *, log: bool) -> None:
        """Point estimates with 95% CIs, one row each; a vertical line marks no effect (1 on a log
        scale, 0 otherwise). Filled marks are replicated estimates, open marks the rest."""
        items = [i for i in items if all(isinstance(v, (int, float)) and math.isfinite(v) for v in i[1:4])
                 and (not log or min(i[1:4]) > 0)]
        if not items:
            self.plain("(no estimates)")
            return
        f = math.log if log else (lambda v: v)
        null = f(1.0) if log else 0.0
        lo_all = min([f(i[2]) for i in items] + [null])
        hi_all = max([f(i[3]) for i in items] + [null])
        span = (hi_all - lo_all) or 1.0
        pdf = self.pdf
        label_w, chart_w, h = 80.0, 70.0, 5.0
        x_of = lambda v: pdf.l_margin + label_w + chart_w * (f(v) - lo_all) / span  # noqa: E731
        pdf.set_font(self.family, "", 7)
        top = pdf.get_y()
        for label, est, lo, hi, replicated in items:
            if pdf.get_y() > pdf.h - 25:
                pdf.add_page()
                top = pdf.get_y()
            y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y)
            pdf.cell(label_w - 2, h, self.text(label[:55]), align="R")
            pdf.set_draw_color(40, 60, 110)
            pdf.line(x_of(lo), y + h / 2, x_of(hi), y + h / 2)
            pdf.set_fill_color(40, 60, 110)
            pdf.rect(x_of(est) - 1, y + h / 2 - 1, 2, 2, style="F" if replicated else "D")
            pdf.set_xy(pdf.l_margin + label_w + chart_w + 1, y)
            pdf.cell(40, h, f"{est:.3g} ({lo:.3g}-{hi:.3g})")
            pdf.ln(h)
        pdf.set_draw_color(150, 150, 150)
        pdf.line(x_of(1.0 if log else 0.0), top, x_of(1.0 if log else 0.0), pdf.get_y())
        pdf.set_draw_color(0, 0, 0)
        pdf.set_fill_color(255, 255, 255)
        pdf.ln(3)


def _forest_items(confirmation: dict[str, Any]) -> tuple[list[tuple], list[tuple]]:
    """(hazard-ratio rows, correlation rows) from a confirmation result: candidates, then positive controls."""
    cox, rank = [], []
    rows = [(r["feature"], r, bool(r.get("replicated"))) for r in confirmation.get("hypotheses") or []]
    rows += [(f"positive control: {r['control']}", r, bool(r.get("detected"))) for r in confirmation.get("positive_controls") or []]
    for label, r, filled in rows:
        ci = r.get("ci95") or [None, None]
        if r.get("test") == "cox":
            cox.append((label, r.get("hr_per_sd"), ci[0], ci[1], filled))
        elif "rho" in r:
            rank.append((label, r.get("rho"), ci[0], ci[1], filled))
    return cox, rank


def build_pdf(out_dir: Path, stem: str, phases: dict[str, Any], events: list[dict[str, Any]],
              coverage_md: str, meta: dict[str, Any]) -> Path:
    """Write ``<stem>_report.pdf`` and return its path.

    ``phases`` is the saved phase output (``<stem>_state.json``), ``events`` the trace events, and
    ``meta`` short run facts for the cover (participants, code version, models).
    """
    doc = _Doc(f"CoDaS report: {stem}")
    pdf = doc.pdf
    pdf.add_page()
    doc.heading(f"CoDaS report: {stem}")
    rounds = phases.get("rounds") or []
    last = rounds[-1] if rounds else {}
    conf = phases.get("confirmation") or {}
    cover = [["target", phases.get("target_column")], ["participants", meta.get("rows")],
             ["discovery rounds", len(rounds)], ["validated candidates (final round)", last.get("validated_count")],
             ["held-out model", f"{last.get('ml_metric_name')} = {last.get('ml_metric_value')}" if last else None],
             *([["replicated on the sealed holdout", f"{conf.get('replicated')} of {conf.get('tested')}"],
                ["also passed falsification", conf.get("robust")]] if conf else []),
             ["code version", meta.get("code_version")], ["models", meta.get("models")],
             ["compiled", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]
    doc.table(None, cover, widths=[60, 120])
    doc.plain("Counts of 1-20 are masked. Every number comes from the deterministic engine; the agents "
              "plan, interpret and write. Review against the Data and Statistics Dissemination Policy "
              "before anything leaves the Workbench.", size=8, style="I")

    table1 = phases.get("cohort_table")
    if table1:
        doc.heading("Cohort characteristics", 2)
        doc.table(table1["columns"], table1["rows"])
        doc.plain(table1.get("note", ""), size=8, style="I")

    pdf.add_page()
    doc.heading("Final report")
    doc.markdown(phases.get("report") or "")

    pdf.add_page()
    doc.heading("Phase A: data understanding and design")
    design = [["target", phases.get("target_column")],
              ["confounders", ", ".join(phases.get("confounder_columns") or []) or "-"],
              ["excluded as leakage", ", ".join(phases.get("excluded_columns") or []) or "-"],
              ["adjustment-only covariates", ", ".join(phases.get("context_columns") or []) or "-"]]
    doc.table(["role", "columns"], design, widths=[50, 130])
    for title, key in (("Exclusions and reasons", "exclusion_log"), ("Confounders declared and reasons", "confounder_log")):
        log = phases.get(key) or []
        if log:
            doc.heading(title, 3)
            doc.table(["after round", "columns", "reason"],
                      [[e.get("after_round"), ", ".join(e.get("columns") or []), e.get("reason")] for e in log],
                      widths=[20, 55, 105])
    doc.heading("Hypotheses and literature anchoring", 2)
    doc.markdown(phases.get("hypotheses") or "")

    pdf.add_page()
    doc.heading("Phases B-C: discovery loop and adversarial validation")
    doc.table(["round", "validated", "evaluated", "held-out model", "above chance", "confounders", "excluded"],
              [[(r["round"] if isinstance(r.get("round"), int) else i) + 1, r.get("validated_count"), r.get("evaluated_count"),
                f"{r.get('ml_metric_name')}={_fmt(r.get('ml_metric_value'))}", r.get("ml_above_chance"),
                ", ".join(r.get("confounders") or []), ", ".join(r.get("excluded_features") or [])]
               for i, r in enumerate(rounds)],
              widths=[13, 18, 18, 35, 18, 39, 39])
    by_round: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        if e.get("round") is not None and e.get("kind") == "text":
            by_round.setdefault(e["round"], []).append(e)
    for r in sorted(by_round):
        doc.heading(f"Round {r}", 2)
        for e in by_round[r]:
            doc.heading(e["author"], 3)
            doc.markdown(e["text"])

    pdf.add_page()
    doc.heading("Phase D: mechanisms and novelty")
    doc.heading("Mechanisms", 2)
    doc.markdown(phases.get("mechanisms") or "")
    doc.heading("Novelty", 2)
    doc.markdown(phases.get("novelty") or "")

    pdf.add_page()
    doc.heading("Phases E-F: strategy and artifacts")
    doc.heading("Strategy", 2)
    doc.markdown(phases.get("strategy") or "")
    doc.heading("Artifacts", 2)
    doc.markdown(phases.get("artifacts") or "")

    pdf.add_page()
    doc.heading("Appendix")
    if conf:
        cox, rank = _forest_items(conf)
        if cox:
            doc.heading("Sealed holdout: hazard ratio per SD (95% CI)", 2)
            doc.forest(cox, log=True)
        if rank:
            doc.heading("Sealed holdout: rank correlation (95% CI)", 2)
            doc.forest(rank, log=False)
        doc.plain("Filled marks replicated (candidates) or were detected (positive controls); open marks did not.",
                  size=8, style="I")
    doc.heading("Candidates in the final round (Spearman rho)", 2)
    doc.bars([(c.get("feature") or "", float(c["rho"])) for c in (last.get("passing_candidates") or [])
              if isinstance(c.get("rho"), (int, float))])
    doc.heading("Phase coverage", 2)
    doc.markdown(coverage_md)
    doc.heading("Fact sheet", 2)
    fact = phases.get("fact_sheet") or {}
    doc.table(["key", "value"], [[k, _fmt(v)] for k, v in fact.items() if not isinstance(v, (dict, list))],
              widths=[80, 100])

    path = out_dir / f"{stem}_report.pdf"
    pdf.output(str(path))
    return path


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)
