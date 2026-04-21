# Judge Threshold Policy

This file defines pass/fail logic for `judge_*.json` outputs produced by benchmark runs.

## Score Gates

Treat a run as passing only when all of these are true:

- `correctness_1_5 >= 4`
- `methodology_ml_1_5 >= 4`
- `leakage_and_validation_1_5 >= 4`

Optional stricter gate:

- `overall_1_5 >= 4`

## Single vs Multi Outcome Label

For each task pair (`single` and `multi`) in a judge result:

- `single_failed_multi_passed`: single fails gates, multi passes gates
- `single_passed_multi_failed`: single passes gates, multi fails gates
- `both_passed`: both pass gates
- `both_failed`: both fail gates

## Recommended Pilot Evaluation

1. Run hard tasks on retail:
   - `retail/14`
   - `retail/15`
   - `retail/16`
2. Run repeated benchmark passes (different `--run-id`) with judge enabled.
3. Aggregate outcomes across runs and track:
   - Multi pass rate
   - Single pass rate
   - `single_failed_multi_passed` rate

Example run command:

`python scripts/run_benchmarks.py --datasets retail_kaggle --tasks retail/14,retail/15,retail/16 --with-judge`
