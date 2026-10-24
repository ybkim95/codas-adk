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
import traceback
from pathlib import Path
from typing import Any

# Font files tried in order: (regular, bold, italic, bold-italic, then the same four for monospace). The
# first family whose regular face exists is used; a missing style falls back to that family's regular face.
# Every style must be registered: Markdown such as **`code`** asks for bold monospace.
_FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/share/fonts/TTF",
              "/usr/local/share/fonts", "/Library/Fonts", str(Path.home() / ".fonts"))
_FAMILIES = (("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf", "DejaVuSans-BoldOblique.ttf",
              "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf",
              "DejaVuSansMono-BoldOblique.ttf"),
             ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf", "NotoSans-Italic.ttf", "NotoSans-BoldItalic.ttf",
              "NotoSansMono-Regular.ttf", "NotoSansMono-Bold.ttf", "NotoSansMono-Regular.ttf",
              "NotoSansMono-Bold.ttf"),
             ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf",
              "LiberationSans-BoldItalic.ttf", "LiberationMono-Regular.ttf", "LiberationMono-Bold.ttf",
              "LiberationMono-Italic.ttf", "LiberationMono-BoldItalic.ttf"))
_STYLES = ("", "B", "I", "BI")
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
            regular, mono = Path(d) / family[0], Path(d) / family[4]
            if regular.exists():
                mono = mono if mono.exists() else regular
                return [Path(d) / f if (Path(d) / f).exists() else (regular if i < 4 else mono)
                        for i, f in enumerate(family)]
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
            for style, path in zip(_STYLES, fonts[:4]):
                self.pdf.add_font("Body", style, str(path))
            for style, path in zip(_STYLES, fonts[4:]):
                self.pdf.add_font("Mono", style, str(path))
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
        page = self.pdf.page
        try:
            self.pdf.set_font(self.family, "", 10)
            self.pdf.write_html(html, font_family=self.family, pre_code_font=self.mono,
                                warn_on_tags_not_matching=False)
        except Exception:  # noqa: BLE001 - a rendering quirk must not lose the content
            # fpdf2's HTML writer sets the current page to 0 while it switches fonts; an error there
            # leaves it at 0, and every later call would fail with "No page open".
            self.pdf.page = self.pdf.page or page
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

    def gantt(self, timeline: list[dict[str, Any]]) -> None:
        """One row per agent and one bar per turn on a shared axis of minutes since the agents started;
        turns that ran in parallel overlap."""
        if not timeline:
            self.plain("(no agent turns recorded)")
            return
        pdf = self.pdf
        agents = list(dict.fromkeys(t["agent"] for t in timeline))
        end = max(t["end_s"] for t in timeline) or 1.0
        label_w, chart_w, h = 45.0, 135.0, 3.6
        x0 = pdf.l_margin + label_w
        pdf.set_font(self.family, "", 6)
        for name in agents:
            if pdf.get_y() > pdf.h - 25:
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_xy(pdf.l_margin, y)
            pdf.cell(label_w - 2, h, self.text(name[:34]), align="R")
            pdf.set_fill_color(42, 120, 214)
            for t in timeline:
                if t["agent"] == name:
                    width = max(chart_w * (t["end_s"] - t["start_s"]) / end, 0.3)
                    pdf.rect(x0 + chart_w * t["start_s"] / end, y + 0.6, width, h - 1.2, style="F")
            pdf.ln(h)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_font(self.family, "", 7)
        pdf.set_x(x0)
        pdf.cell(chart_w, 4, self.text(f"0 to {end / 60:.1f} minutes after the agents started"), align="C")
        pdf.ln(6)


def _analysis_figures(analyses: list[dict[str, Any]]) -> list[tuple[str, list[tuple], bool]]:
    """Figures from the recorded analyses: subgroup forests, the dose-response curve and the importance of
    the first ML model (title, items, log scale)."""
    figures: list[tuple[str, list[tuple], bool]] = []
    for a in analyses:
        result, tool = a.get("result") or {}, str(a.get("tool", ""))
        if "error" in result:
            continue
        subgroup_sets = ([result] if tool == "subgroup_analysis" else
                         result.get("results") or [] if tool == "method:subgroup_stratified" else [])
        for sub in subgroup_sets[:2]:
            strata = sub.get("strata") or []
            cox = any("hr_per_sd" in st for st in strata)
            items = [(st["stratum"], st.get("hr_per_sd") if cox else st.get("rho"), *(st.get("ci95") or [None, None]), False)
                     for st in strata]
            figures.append((f"Subgroups: {sub.get('feature')} by {sub.get('by')} (heterogeneity p "
                            f"{_fmt(sub.get('heterogeneity_p'))})", items, cox))
        if tool == "dose_response":
            log = result.get("model") != "ols"
            items = [(f"{c['percentile']}th percentile ({_fmt(c['value'])})", c["effect"], *c["ci95"], False)
                     for c in result.get("curve") or [] if c["percentile"] != 50]
            figures.append((f"Dose-response: {result.get('feature')} ({result.get('measure')}, non-linearity p "
                            f"{_fmt(result.get('nonlinearity_p'))})", items, log))
    ml = next((a for a in analyses if str(a.get("tool", "")).startswith("method:") and
               (a.get("result") or {}).get("importance")), None)
    if ml:
        res = ml["result"]
        figures.append((f"Feature importance: {res.get('method')} (held-out {res.get('metric')} {_fmt(res.get('value'))})",
                        [(r["feature"], float(r["importance"])) for r in res["importance"][:12]], False))
    return figures[:8]


def _forest_items(confirmation: dict[str, Any]) -> tuple[list[tuple], list[tuple]]:
    """(hazard-ratio rows, correlation rows) from a confirmation result: candidates, then positive controls."""
    cox, rank = [], []
    rows = [(r["feature"], r, bool(r.get("replicated") or r.get("reproduced"))) for r in confirmation.get("hypotheses") or []]
    rows += [(f"positive control: {r['control']}", r, bool(r.get("detected"))) for r in confirmation.get("positive_controls") or []]
    for label, r, filled in rows:
        ci = r.get("ci95") or [None, None]
        if r.get("test") == "cox":
            cox.append((label, r.get("hr_per_sd"), ci[0], ci[1], filled))
        elif "rho" in r:
            rank.append((label, r.get("rho"), ci[0], ci[1], filled))
    return cox, rank


def build_pdf(out_dir: Path, stem: str, phases: dict[str, Any], events: list[dict[str, Any]],
              coverage_md: str, meta: dict[str, Any], execution: dict[str, Any] | None = None) -> Path:
    """Write ``<stem>_report.pdf`` and return its path.

    ``phases`` is the saved phase output (``<stem>_state.json``), ``events`` the trace events, and
    ``meta`` short run facts for the cover (participants, code version, models), and ``execution`` the
    agent execution tables (``codas.aou.execution``), shown in their own section. Each section is
    rendered on its own: one that fails is replaced by a note, the rest of the report is kept, and the
    full traceback goes to ``<stem>_pdf_error.txt`` (removed again when every section renders).
    """
    doc = _Doc(f"CoDaS report: {stem}")
    pdf = doc.pdf
    rounds = phases.get("rounds") or []
    last = rounds[-1] if rounds else {}
    conf = phases.get("confirmation") or {}

    def cover() -> None:
        doc.heading(f"CoDaS report: {stem}")
        rows = [["target", phases.get("target_column")], ["participants", meta.get("rows")],
                ["discovery rounds", len(rounds)], ["validated candidates (final round)", last.get("validated_count")],
                ["held-out model", f"{last.get('ml_metric_name')} = {_fmt(last.get('ml_metric_value'))}" if last else None],
                *([["replicated on the sealed holdout", f"{conf.get('replicated')} of {conf.get('tested')}"],
                   ["also passed falsification", conf.get("robust")]] if conf else []),
                ["code version", meta.get("code_version")], ["models", meta.get("models")],
                ["compiled", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")]]
        doc.table(None, rows, widths=[60, 120])
        doc.plain("Counts of 1-20 are masked. Every number comes from the deterministic engine; the agents "
                  "plan, interpret and write. Review against the Data and Statistics Dissemination Policy "
                  "before anything leaves the Workbench.", size=8, style="I")

    def cohort() -> None:
        table1 = phases.get("cohort_table")
        if table1:
            doc.heading("Cohort characteristics", 2)
            doc.table(table1["columns"], table1["rows"])
            doc.plain(table1.get("note", ""), size=8, style="I")

    def final_report() -> None:
        doc.heading("Final report")
        doc.markdown(phases.get("report") or "")

    def data_understanding() -> None:
        dossier, signal = phases.get("eda") or {}, phases.get("signal_map") or {}
        rules, units = phases.get("quality_rules") or [], (phases.get("extraction_qc") or {}).get("units") or {}
        if not dossier and not signal and not rules and not units.get("units"):
            return
        doc.heading("Data understanding")
        if rules:
            doc.heading("Quality rules applied at extraction", 2)
            for rule in rules:
                doc.plain(f"- {rule}", size=8)
        if units.get("units"):
            doc.heading("Units of the outcome's values (discovery share)" + (": more than one scale" if units.get("mixed")
                                                                            else ""), 2)
            doc.table(["unit", "concept id", "share", "p25", "median", "p75"],
                      [[u.get("unit"), u.get("unit_concept_id"), u.get("share"), u.get("p25"), u.get("median"), u.get("p75")]
                       for u in units["units"]], widths=[60, 25, 35, 20, 20, 20])
        if dossier:
            doc.plain("Read by the agents before any hypothesis was registered; it never relates a feature to the "
                      "outcome. Counts of 1-20 and quantiles with fewer than 20 people beyond them are not shown.",
                      size=8, style="I")
            feats = dossier.get("features") or {}
            flagged = {f: d for f, d in feats.items() if d.get("near_constant") or
                       (isinstance(d.get("skew"), float) and abs(d["skew"]) > 2) or
                       (isinstance(d.get("missing"), float) and d["missing"] > 0.2)}
            doc.heading(f"Candidate features ({len(feats)}; {len(flagged)} need care)", 2)
            reliability = dossier.get("reliability") or {}
            doc.table(["feature", "n", "missing", "mean", "SD", "skew", "beyond 4 SD", "natural unit", "split-half",
                       "note"],
                      [[f, d.get("n"), d.get("missing"), d.get("mean"), d.get("sd"), d.get("skew"), d.get("beyond_4sd"),
                        d.get("natural_unit"), (reliability.get(f) or {}).get("reliability"),
                        "near-constant" if d.get("near_constant") else d.get("note", "")]
                       for f, d in feats.items()], widths=[42, 13, 16, 17, 15, 13, 16, 17, 15, 16])
            if dossier.get("families"):
                doc.heading("Feature families (members correlate at |rho| >= 0.7)", 2)
                doc.table(["representative", "members", "|rho| range"],
                          [[g["representative"], ", ".join(g["members"]), f"{g['min_abs_rho']}-{g['max_abs_rho']}"]
                           for g in dossier["families"]], widths=[45, 110, 25])
            wear = dossier.get("wear_dependence") or {}
            if wear:
                doc.heading("Wear dependence (strongest |rho| with a days-worn column)", 2)
                doc.table(["feature", "|rho|", "with"], [[f, d["max_abs_rho"], d["with"]] for f, d in
                                                         sorted(wear.items(), key=lambda kv: -kv[1]["max_abs_rho"])[:25]],
                          widths=[80, 20, 80])
            miss = dossier.get("missingness") or {}
            if miss:
                doc.heading("Who is missing each feature (standardised mean difference, missing vs present)", 2)
                keys = sorted({k for d in miss.values() for k in d if k.startswith("smd_")})
                doc.table(["feature", "missing", *[k[4:] for k in keys]],
                          [[f, d.get("missing"), *[d.get(k) for k in keys]] for f, d in miss.items()])
            outcome = dossier.get("outcome") or {}
            if outcome:
                doc.heading("The outcome on its own", 2)
                rows = [["overall", *_outcome_cells(outcome.get("overall") or {}, outcome.get("kind"))]]
                rows += [[k, *_outcome_cells(v, outcome.get("kind"))] for k, v in (outcome.get("by_stratum") or {}).items()]
                header = (["stratum", "participants", "events", "person-years", "per 1,000 PY", "censored"]
                          if outcome.get("kind") == "time to event" else ["stratum", "n", "mean", "", "", ""])
                doc.table(header, rows)
        if signal.get("features"):
            doc.heading("Signal map (after registration closed)", 2)
            doc.plain(f"Discovery share. T0 adjusts for {', '.join(signal.get('base') or []) or 'nothing'}; T2 also "
                      f"for {', '.join(signal.get('covariates') or []) or 'nothing more'}. Attenuation is the share "
                      "of the T0 effect the covariates explain; q-values correct across features.", size=8, style="I")
            rows = sorted(signal["features"], key=lambda r: (r.get("q_t2") if r.get("q_t2") is not None else 2))
            doc.table(["feature", "measure", "T0", "T2", "q (T2)", "attenuation", "missing-indicator effect"],
                      [[r["feature"], r["measure"], r.get("t0"), r.get("t2"), r.get("q_t2"), r.get("attenuation"),
                        r.get("missing_effect")] for r in rows[:40]], widths=[50, 30, 18, 18, 18, 22, 24])

    def literature() -> None:
        matrix, audit = phases.get("literature_evidence") or {}, phases.get("citations") or {}
        if not matrix and not audit.get("used"):
            return
        doc.heading("Literature evidence")
        status = phases.get("literature_status") or "status not recorded"
        doc.plain(f"Counted in Europe PMC before the agents started ({status}): papers naming the construct and the "
                  "outcome in the title or abstract, the reviews and "
                  "meta-analyses among them, and the earliest year. The class is counted, not judged: established (at "
                  "least one review or ten papers), emerging (fewer), unexplored (none), not assessed (no answer).",
                  size=8, style="I")
        columns_of: dict[str, list[str]] = {}
        for column, construct in (phases.get("construct_of") or {}).items():
            columns_of.setdefault(construct, []).append(column)
        doc.table(["construct", "columns", "class", "papers", "reviews", "earliest", "most cited"],
                  [[c, ", ".join(columns_of.get(c, [])[:4]), e.get("novelty"), e.get("papers"), e.get("reviews"),
                    e.get("earliest_year"), "; ".join(p["key"] for p in (e.get("top_cited") or [])[:2])]
                   for c, e in matrix.items()], widths=[40, 42, 20, 14, 14, 14, 36])
        if audit.get("used"):
            doc.plain(f"Citation keys in the agents' text: {len(audit['used'])}; {len(audit.get('verified') or [])} among the "
                      f"{audit.get('retrieved')} papers retrieved in this run"
                      + (f"; not citations: {', '.join(audit['unverified'])}" if audit.get("unverified") else "") + ".",
                      size=8)

    def phase_a() -> None:
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
        from codas.core.hypotheses import RUBRIC

        audit = phases.get("data_memo_audit") or {}
        if audit.get("claims"):
            doc.heading("The data analyst's memo (each claim checked against the data summary)", 2)
            doc.plain(audit.get("summary") or "", size=9)
            doc.table(["claim", "rests on", "check", "implication"],
                      [[c["statement"], ", ".join(c.get("fields") or []),
                        c["status"] + (f": {c['detail']}" if c.get("detail") else ""), c.get("implication")]
                       for c in audit["claims"]], widths=[60, 40, 35, 45])
        defined = phases.get("defined_phenotypes") or []
        if defined:
            doc.heading("Phenotypes defined on the day-by-day baseline", 2)
            doc.table(["column", "computes", "rationale"],
                      [[d["column"], d.get("description"), d.get("rationale")] for d in defined], widths=[45, 80, 55])
        doc.heading("Hypotheses and literature anchoring", 2)
        doc.markdown(phases.get("hypotheses") or "")
        registered = phases.get("a_priori_hypotheses") or []
        if registered:
            doc.heading("Hypotheses registered before any statistic (tested on the sealed holdout)", 3)
            doc.table(["feature", "predicted", "knownness (evidence)", "declared", "construct", "citation"],
                      [[h.get("feature"), "positive" if h.get("direction", 0) > 0 else "negative", h.get("knownness"),
                        h.get("declared_knownness"), h.get("construct"), h.get("citation")] for h in registered],
                      widths=[44, 18, 22, 18, 38, 40])
            doc.heading("Each hypothesis as designed: what it must beat, its shape, magnitude and predictions", 3)
            doc.table(["feature", "incumbent", "shape", "magnitude", "source", "predictions", "mechanism"],
                      [[h.get("feature"), h.get("incumbent") or "-", h.get("shape"), _magnitude_text(h.get("magnitude")),
                        (h.get("magnitude") or {}).get("source"), ", ".join(h.get("predictions") or []),
                        " -> ".join(h.get("mechanism_steps") or [])] for h in registered],
                      widths=[30, 25, 16, 22, 18, 35, 34])
        refused = [a for a in phases.get("hypothesis_attempts") or [] if not a.get("accepted")]
        if refused:
            doc.heading("Submissions refused at registration (corrected and resubmitted, or dropped)", 3)
            doc.table(["feature", "parts", "reasons"],
                      [[a.get("feature"), ", ".join(dict.fromkeys(a.get("parts") or [])), "; ".join(a.get("problems") or [])]
                       for a in refused], widths=[40, 30, 110])
        review = phases.get("hypothesis_review") or []
        if review:
            doc.heading("Independent review (1-5; secondary evidence, since a model rates its own kind unreliably)", 3)
            doc.table(["feature", *[k.replace("_", " ") for k in RUBRIC], "comment"],
                      [[r["feature"], *[r["scores"].get(k) for k in RUBRIC], r.get("comment")] for r in review],
                      widths=[30, 14, 16, 16, 14, 16, 16, 58])

    def phases_bc() -> None:
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
        ran = phases.get("analyses") or []
        if ran:
            from codas.core.methods import record_headline

            doc.heading("Methods and analyses run (method pools and agents' tools)", 2)
            doc.table(["round", "analysis", "arguments", "result"],
                      [[_fmt(a.get("round")), a.get("tool"),
                        ", ".join(f"{k}={v}" for k, v in (a.get("args") or {}).items() if v),
                        record_headline(a)] for a in ran], widths=[12, 32, 30, 106])
        gaps = phases.get("gap_analysis") or []
        if gaps:
            doc.heading("Gap analyses (what each round left open, and what ran next)", 2)
            doc.table(["after round", "open questions", "next methods"],
                      [[_fmt(g.get("round")), "; ".join(g.get("gaps") or []),
                        ", ".join(m.get("method", "") + "".join(f" {k}={v}" for k, v in m.items() if k != "method")
                                  for m in g.get("next_methods") or [])] for g in gaps], widths=[18, 102, 60])

    def phase_d() -> None:
        doc.heading("Phase D: per-candidate debate, mechanisms and novelty")
        debates = phases.get("candidate_debates") or []
        if debates:
            doc.heading("Per-candidate debate (critic, then defender, for each final candidate)", 2)
            doc.table(["candidate", "verdict", "reason", "evidence"],
                      [[d.get("feature"), d.get("verdict"), d.get("reason"), d.get("evidence")] for d in debates],
                      widths=[45, 18, 87, 30])
        doc.heading("Mechanisms", 2)
        doc.markdown(phases.get("mechanisms") or "")
        doc.heading("Novelty", 2)
        doc.markdown(phases.get("novelty") or "")

    def phases_ef() -> None:
        doc.heading("Phases E-F: strategy, artifacts, report review and lessons")
        doc.heading("Strategy", 2)
        doc.markdown(phases.get("strategy") or "")
        doc.heading("Artifacts", 2)
        doc.markdown(phases.get("artifacts") or "")
        if phases.get("report_review"):
            doc.heading("Report review (issues the reviewer raised; the final report is the revision)", 2)
            doc.markdown(phases.get("report_review") or "")
        lessons = phases.get("lessons") or []
        if lessons:
            doc.heading("Lessons recorded for later outcomes on this cohort", 2)
            doc.table(["lesson", "about", "evidence"],
                      [[x.get("lesson"), ", ".join(x.get("about") or []), x.get("evidence")] for x in lessons],
                      widths=[100, 40, 40])

    def agent_execution() -> None:
        if not (execution or {}).get("agents"):
            return
        from codas.aou.execution import totals_text

        doc.heading("Agent execution")
        doc.plain(totals_text(execution.get("totals") or {}), size=9)
        doc.table(["agent", "turns", "active min", "model calls", "errors", "tool calls (by tool)", "tokens",
                   "thinking", "models served", "cost (USD)"],
                  [[a["agent"], a["turns"], f"{a['active_s'] / 60:.1f}", a["model_calls"],
                    a["model_errors"] + a["tool_errors"],
                    f"{a['tool_calls']} " + ", ".join(f"{k} x{v}" for k, v in sorted(a["tools"].items(),
                                                                                     key=lambda kv: -kv[1])),
                    f"{a['total']:,}", f"{a['thoughts']:,}", ", ".join(f"{k} x{v}" for k, v in a["models"].items()),
                    "" if a["cost_usd"] is None else f"{a['cost_usd']:.3f}"] for a in execution["agents"]],
                  widths=[28, 10, 12, 12, 10, 42, 16, 14, 24, 12])
        if execution.get("timeline"):
            doc.heading("Agent turns over time", 2)
            doc.gantt(execution["timeline"])
        if execution.get("design_changes"):
            doc.heading("Design changes (who changed what the later agents work on, and in which round)", 2)
            doc.table(["round", "minute", "agent", "tool"],
                      [[c["round"], f"{c['at_s'] / 60:.1f}", c["agent"], c["tool"] + (" (error)" if c["error"] else "")]
                       for c in execution["design_changes"]], widths=[15, 20, 65, 80])
        if execution.get("flow"):
            from codas.aou.execution import flow_by_writer

            doc.heading("Information flow (a memory key one agent wrote, and the later agents that read it)", 2)
            doc.table(["writer", "key", "read by (turns)"],
                      [[f["writer"], f["key"], f["read by (turns)"]] for f in flow_by_writer(execution["flow"])],
                      widths=[38, 38, 104])

    def evidence() -> None:
        card = phases.get("scorecard") or {}
        doc.heading("What the analysis covered, and how far the evidence reaches")
        doc.table(["", "measure", "value"], [[s["section"], s["measure"], s["value"]] for s in card.get("summary") or []],
                  widths=[30, 95, 55])
        records = card.get("hypotheses") or []
        if records:
            from codas.aou.scorecard import STEPS

            doc.heading("Recipe coverage and the evidence ladder", 2)
            doc.plain("Rung 1: associated on the sealed holdout; 2: replicated and robust to every falsification test "
                      "that ran; 3: adds predictive value over the covariates.", size=8, style="I")
            rows = []
            for r in records:
                done = [label for key, label in STEPS if r["cells"][key][0] == "yes"]
                missing = [f"{label} ({r['cells'][key][1]})" for key, label in STEPS if r["cells"][key][0] == "no"]
                rows.append([r["feature"], f"{r['rung']}: {r['rung_label']}", "; ".join(done), "; ".join(missing) or "-",
                             r["next"]])
            doc.table(["feature", "rung", "ran", "missing", "next step"], rows, widths=[32, 30, 50, 40, 28])
        varied = [(h["feature"], e) for h in (conf or {}).get("hypotheses") or [] for e in h.get("heterogeneity") or []
                  if e.get("q_p") is not None]
        if varied:
            doc.heading("Sealed holdout: consistency across strata (effects only; Q test and I^2)", 2)
            doc.table(["feature", "across", "stratum effects", "Q p", "I^2"],
                      [[f, e["by"], "; ".join(f"{s['stratum']}: {s.get('hr_per_sd', s.get('rho'))}" if "ci95" in s
                                                else f"{s['stratum']}: {s.get('status')}" for s in e["strata"]),
                        _fmt(e["q_p"]), _fmt(e["i2"])] for f, e in varied], widths=[40, 25, 85, 15, 15])

    def appendix() -> None:
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
        for title, items, log in _analysis_figures(phases.get("analyses") or []):
            doc.heading(title, 2)
            if items and isinstance(items[0][1], (int, float)) and len(items[0]) == 2:
                doc.bars(items)
            else:
                doc.forest(items, log=log)
        doc.heading("Candidates in the final round (Spearman rho)", 2)
        doc.bars([(c.get("feature") or "", float(c["rho"])) for c in (last.get("passing_candidates") or [])
                  if isinstance(c.get("rho"), (int, float))])
        doc.heading("Phase coverage", 2)
        doc.markdown(coverage_md)
        doc.heading("Fact sheet", 2)
        fact = phases.get("fact_sheet") or {}
        doc.table(["key", "value"], [[k, _fmt(v)] for k, v in fact.items() if not isinstance(v, (dict, list))],
                  widths=[80, 100])

    # (section, starts on a new page)
    sections = [(cover, True), (cohort, False), (final_report, True),
                (data_understanding, bool(phases.get("eda") or phases.get("signal_map"))),
                (literature, bool(phases.get("literature_evidence"))), (phase_a, True),
                (phases_bc, True),
                (phase_d, True), (phases_ef, True),
                (agent_execution, bool((execution or {}).get("agents"))),
                (evidence, bool((phases.get("scorecard") or {}).get("summary"))), (appendix, True)]
    errors = []
    for section, new_page in sections:
        if new_page or not pdf.page:
            pdf.add_page()
        try:
            section()
        except Exception as exc:  # noqa: BLE001 - one section's failure must not lose the whole report
            errors.append(f"--- section {section.__name__} ---\n{traceback.format_exc()}")
            pdf.page = pdf.page or len(pdf.pages)
            pdf.set_fill_color(255, 255, 255)
            doc.plain(f"[The {section.__name__.replace('_', ' ')} section could not be rendered: "
                      f"{type(exc).__name__}: {str(exc)[:200]}]", style="I")
    error_file = out_dir / f"{stem}_pdf_error.txt"
    if errors:
        from codas.aou.privacy import mask_identifiers

        error_file.write_text(mask_identifiers("\n".join(errors)), encoding="utf-8")
    else:
        error_file.unlink(missing_ok=True)
    path = out_dir / f"{stem}_report.pdf"
    pdf.output(str(path))
    return path


def _magnitude_text(m: dict[str, Any] | None) -> str:
    if not m or not isinstance(m.get("low"), (int, float)) or not isinstance(m.get("high"), (int, float)):
        return ""
    return f"{m.get('measure')} {m['low']:g}-{m['high']:g}"


def _outcome_cells(summary: dict[str, Any], kind: str | None) -> list[Any]:
    if kind == "time to event":
        return [summary.get("participants"), summary.get("events"), summary.get("person_years"),
                summary.get("per_1000_person_years"), summary.get("censored")]
    return [summary.get("n"), summary.get("mean"), "", "", ""]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)
