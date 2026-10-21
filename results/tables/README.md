# Tables

Flat CSVs for drawing your own figures, written by scripts/export_tables.py from the summaries under results/.

- all_summaries.csv: every value of every summary.json (run, model, experiment, key path, value, and the 95% scene-bootstrap interval where the summary carries one). Grep the key path for what you need, for example `arms/all heads of 7+11/names_X` or `questions/animal/from/11/cost`.
- knockout_costs.csv: attention knockouts, the cost in nats of blocking one layer (single), every layer from L on (from) or up to L (upto), per model and question, with intervals.
- knockout_stations.csv: per model and question, the readers (layers whose block costs 1 nat or more), the last reader, the clean log-probability difference and what blocking every layer leaves.
- battery_depths.csv and battery_curves.csv: single-layer patching with log-probabilities (read_battery), the read depth per question by flips and by log-probability, and the full curve over patch layers.
- access_profiles.csv: per model and question class, the depth by which half the read is done, the window width in depth and in layers, and the largest share any single layer carries.

Per-scene answers for every run are the gzipped CSVs under results/oct21/<run>/<model>/ (pandas reads .csv.gz directly); the 14 October per-scene outputs are under results/<experiment>/<model>/.
