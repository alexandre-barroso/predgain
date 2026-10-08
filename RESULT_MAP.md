# Result and verification map

| Claim | Reproduction output or evidence | Scope |
|---|---|---|
| EN-A1 | `results.json` → `scores`; model manifests and original score caches | 1,726 released scores per checkpoint, target-ID substitution, canonical-path change, 0.001-nat tolerance |
| EN-A2 | `results.json` → `order`; keyed-lag synthetic invariant | 1,521 row-order predecessors, with 165 immediate, 575 earlier nonadjacent and 781 later |
| EN-A3 | `results.json` → `observations` | Four aggregate eye-time fields, represented readers and first-entry eligibility; no missing-trial imputation |
| EN-A4 | `analysis_frame`, `word_annotations`, analysis cell counts | 761 eligible content targets; 1,104/510 common lag-eligible rows |
| EN-A6 | All 16 `analyses` → `partition_runs` and `fixed_prediction_strata` | Fixed-form OLS on 20 sentence partitions; all eight flagged content strata must retain negative MSE decreases |
| EN-A7 | `deletion_refits`, `fixed_prediction_sentence_deletion_mse_decrease_range` | Reduced-sample refit-and-resplit diagnostics versus omissions of evaluated losses |
| EN-A8 | Pinned public snapshots linked in README and manuscript | Documentary history; not established by model fitting or by absent issue reports |
| Incremental score value | `results.json` → `incremental_score.baselines` and `.comparisons`; five `en_incremental_*.csv` exports | 320 distinct baselines and 320 four-block comparisons; complete N/F/I losses and normalized values; 160/160 positive all-word changes, 58/160 negative content changes, and all negative I retained |
| Position/predecessor block dependence | `incremental_score.comparisons` → `N_paths`, `F_paths`, `I_paths`; `en_incremental_paths.csv` | Both paths and defined nonadditivity; block operations include normalization/interactions or lagged lexical/score terms, not causal effects |
| Extension computational coverage | `incremental_score.diagnostic_artifacts`; bound fold files | 1,600 nuisance + 6,400 full fits; all 200 sentence-partition membership hashes; exact common masks and original endpoint agreement within 10^-9 ms² |

The prior exploratory exponent, boundary-correction and response-link comparisons are outside this frozen scientific core. This package does not promote them to confirmatory evidence. All substantive analyses were exploratory; the fixed code specification does not retrospectively preregister them.

The matched-baseline extension follows the earlier exploratory first-fixation component fits. Its new comparisons preserve the same two outcomes, two score constructions, two checkpoints, common masks and twenty partitions. A baseline is fitted once per selection/outcome/seed/block and reused across model/score labels. Repeated partitions and repeated baseline display are not independent replications. Verification preserves the prior joint, stratum and deletion checks before computing the extension.
