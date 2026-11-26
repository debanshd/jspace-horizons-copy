# J++ post evals: omni J-lenses and Neuronpedia baselines

jpp_lens d7efc0d. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at

the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary

(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.

| model | lens | filtered_macro_r@1 | filtered_macro_r@10 | multihop_r@10 | multilingual_r@10 | typo_r@10 | association_r@10 | poetry_r@10 | full_macro_r@1 | full_macro_r@10 | swap_top1 | swap_top1_hits | swap_top5 | swap_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmo-3-7b | logit | 13.0 | 33.7 | 64.7 | 46.3 | 41.6 | 6.5 | 9.5 | 13.0 | 33.4 | 16.3 | 7.0/43.0 | 54.8 | 2.0 |
| olmo-3-7b | np_olmo-3-7b_jlens | 13.9 | 37.7 | 64.7 | 49.4 | 57.1 | 7.8 | 9.5 | 12.8 | 33.7 | 39.5 | 17.0/43.0 | 71.0 | 1.0 |
| olmo-3-7b | olmo7b_jpp_n650 | 29.9 | 52.2 | 82.4 | 48.6 | 96.1 | 19.5 | 14.3 | 21.3 | 46.1 | 48.8 | 21.0/43.0 | 67.7 | 1.0 |
| olmo-3-7b-50pct | logit | 8.9 | 27.9 | 70.8 | 33.3 | 23.4 | 6.5 | 5.3 | 8.9 | 27.9 | 9.3 | 4.0/43.0 | 57.6 | 2.0 |
| olmo-3-7b-50pct | olmo7b50pct_jpp_n625 | 21.3 | 39.7 | 50.0 | 28.8 | 90.9 | 18.2 | 10.5 | 20.2 | 36.7 | 41.9 | 18.0/43.0 | 75.8 | 1.0 |

Wall time per step, seconds:

- olmo-3-7b: fetch_lenses 4, readout_filtered 14, readout_full 14, probe_swap 33
- olmo-3-7b-50pct: fetch_lenses 0, readout_filtered 13, readout_full 13, probe_swap 27
