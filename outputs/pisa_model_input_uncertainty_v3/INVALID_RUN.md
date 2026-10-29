# Invalid run notice

This v3 run is invalid and must not be used for analysis or reporting.

The panel contains one row per country-cycle-domain, with all three PISA score
columns repeated across those rows. The script constructed each score series
without filtering to the matching `domain`, duplicating country-cycle values
as artificial repeated observations in context and training data.

Five 25-draw checkpoint batches (draws 0–124) were produced before the issue
was detected. The next batch was interrupted. These outputs are retained only
for auditability; they are not valid results. The corrected run is implemented
in `scripts/run_pisa_model_input_uncertainty_v4.py` and writes to the separate
`outputs/pisa_model_input_uncertainty_v4/` directory.
