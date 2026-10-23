# Incident register

Every failure seen in a real All of Us run, or found in review, with its root cause and the test that now
guards it. A fix is not finished until its row names a test that would have caught the failure. The
integrity rows (19–22) are checked on every milestone.

| # | Where seen | What went wrong | Root cause | Guard (test) |
|---|---|---|---|---|
| 1 | run 2 (Phase 13) | Agents crashed inside Jupyter | `asyncio.run` inside a running event loop | `test_aou_e2e.py::test_run_agents_inside_a_running_event_loop` |
| 2 | run 4 (Phase 17) | One 429 from Vertex ended the whole exploration | No fallback model; one outcome's failure was fatal | `test_resilience.py::test_rate_limited_model_falls_back_to_the_next_one`, `::test_a_failing_outcome_does_not_stop_the_others` |
| 3 | run 5 (Phase 18) | Diagnosed and undiagnosed people had index dates set by different rules | The non-case index date was anchored to the last wear day | `test_aou_sql.py::test_incident_condition_cohort_matches_hand_computed_rows` |
| 4 | run 6 (Phase 23) | Every extraction failed with an INT64 overflow | Holdout hash multiplied 12-digit person ids | `test_aou_sql.py::test_holdout_split_handles_all_of_us_sized_person_ids` |
| 5 | runs 5 and 7 (Phase 25) | A stale notebook page ran old code | The page was never reloaded after a pull | `test_job.py::test_the_notebook_stops_when_the_page_is_an_older_copy` |
| 6 | v6 (Phase 26) | No PDF for any outcome | One failing section lost the whole file | `test_pdf_report.py::test_a_failing_section_keeps_the_rest_of_the_report` |
| 7 | v7 (Phase 27) | Tables and figures cut off in print | Unbreakable identifiers; fixed-size figures | `test_notebook_views.py::test_tables_wrap_long_identifiers_at_underscores_only`, `::test_charts_scale_to_the_page` |
| 8 | v8 (Phase 30) | A 288-page notebook PDF | A library warning re-printed in a loop | `test_job.py::test_a_flood_of_one_line_becomes_a_few_lines_and_a_count`, `::test_warnings_show_once_even_when_other_code_resets_the_registry` |
| 9 | v8 (Phase 31) | Four outcomes crashed after their agents finished | `math.log(0)` in the Cox partial likelihood on an overshooting step | `test_survival.py::test_an_overshooting_newton_step_is_halved_instead_of_failing` |
| 10 | v8 (Phase 31) | About 7 h of agent work lost | Outputs saved only after confirmation | `test_resilience.py::test_agent_work_survives_a_failure_after_the_agents` |
| 11 | v8 (Phase 32) | Every negative-control Cox was `nan`, and `nan` counted as passed | Two identical covariates made the model singular; `nan` read as "no association" | `test_survival.py::test_a_duplicated_or_constant_covariate_is_dropped_not_fatal`, `test_confirm.py::test_negative_controls_that_could_not_be_computed_never_pass`, `test_preflight.py::test_preflight_stops_on_checks_that_cannot_be_computed` |
| 12 | v8 (Phase 32) | The same analysis repeated dozens of times | No record of what had already run | `test_analysis_tools.py::test_a_repeated_analysis_returns_the_earlier_result_without_recomputing` |
| 13 | v8 re-run (Phase 33) | A protective effect (HR 0.85, q 5e-5) was judged "not replicated" | The pre-registered direction came from a screen that ignores follow-up | `test_confirm.py::test_survival_direction_comes_from_the_discovery_cox_model` |
| 14 | v8 re-run (Phase 33) | The wear-adherence check tested nothing | Its only column duplicated a confounder | `test_confirm.py::test_adherence_check_is_not_counted_when_wear_is_already_adjusted_for`, `::test_duplicate_availability_columns_are_used_once` |
| 15 | v8 re-run (Phase 33) | Reports contradicted the holdout | The narrative is written before the holdout opens; it saw only the latest analyses | `test_confirm.py::test_reconciliation_puts_engine_debate_and_holdout_side_by_side`, `test_analysis_tools.py::test_every_analysis_run_is_counted_for_the_report_writers` |
| 16 | v8 re-run (Phase 33) | About 13 TB of minute tables read per outcome | Phenotypes recomputed for every outcome | `test_aou_sql.py::test_phenotypes_are_read_once_per_run_and_match_the_inline_extraction` |
| 17 | v8 re-run (Phase 33) | Only 27 of 70 literature searches returned sources | No retry on 429 | `test_agent_loop.py::test_grounded_search_retries_when_rate_limited` |
| 18 | v8 re-run (Phase 33) | eGFR had no participants | Only one formula's code; one missing code failed the whole item | `test_concept_lookup.py::test_codes_missing_from_this_vocabulary_are_left_out_and_reported` |
| 19 | review (Phase 35) | Hypotheses registered "a priori" could have seen the outcome association | `propose_feature` returned the feature's ρ with the outcome before registration | `test_integrity.py::test_nothing_the_agents_see_before_registration_depends_on_the_outcome`, `::test_propose_feature_withholds_the_outcome_association_until_registration_closes` |
| 20 | review (Phase 35) | Lessons from other outcomes reached the a-priori stage | Lessons were written into the goal every agent reads | `test_resilience.py::test_lessons_from_one_outcome_reach_the_next_without_changing_resume_keys` |
| 21 | review (Phase 35) | The holdout could be re-spent | Every code change re-registered the same outcomes; q was recomputed per registration | `test_confirm.py::test_ledger_never_lets_a_rerun_or_a_known_outcome_count_again`, `::test_lord_levels_are_valid_and_reward_discoveries`, `::test_established_hypotheses_form_a_reference_family_and_old_entries_open_their_outcomes` |
| 22 | review (Phase 35) | The sealed share was reachable by path | The guardrail allowed the whole runs folder; tools accepted any path | `test_integrity.py::test_agents_never_open_the_sealed_share`, `::test_the_guardrail_refuses_sealed_paths_and_aou_tools_ignore_agent_paths` |
| 23 | review (Phase 35) | A runaway query could bill without limit | No dry run and no byte cap | `test_aou.py::test_bigquery_jobs_over_the_cost_caps_are_refused_before_they_run` |
