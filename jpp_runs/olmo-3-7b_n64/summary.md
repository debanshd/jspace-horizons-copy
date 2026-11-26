# J++ post evals: omni J-lenses and Neuronpedia baselines

jpp_lens d7efc0d. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at

the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary

(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.

| model | lens | filtered_macro_r@1 | filtered_macro_r@10 | multihop_r@10 | multilingual_r@10 | typo_r@10 | association_r@10 | poetry_r@10 | full_macro_r@1 | full_macro_r@10 | swap_top1 | swap_top1_hits | swap_top5 | swap_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmo-3-7b | logit | 12.4 | 33.0 | 64.7 | 46.3 | 37.7 | 6.5 | 10.0 | 12.4 | 32.8 | 16.3 | 7.0/43.0 | 54.8 | 2.0 |
| olmo-3-7b | np_olmo-3-7b_jlens | 13.9 | 37.6 | 64.7 | 49.4 | 55.8 | 7.8 | 10.0 | 12.5 | 33.0 | 39.5 | 17.0/43.0 | 71.0 | 1.0 |
| olmo-3-7b | olmo7b_jpp_n64 | 27.4 | 53.4 | 88.2 | 54.6 | 88.3 | 26.0 | 10.0 | 24.2 | 45.9 | 53.5 | 23.0/43.0 | 74.2 | 1.0 |
| olmo-3-7b-50pct | logit | 8.9 | 28.1 | 70.8 | 33.3 | 24.7 | 6.5 | 5.3 | 8.6 | 28.1 | 9.8 | 4.0/41.0 | 54.8 | 2.0 |
| olmo-3-7b-50pct | olmo7b50pct_jpp_n64 | 21.2 | 39.5 | 58.3 | 26.3 | 83.1 | 24.7 | 5.3 | 18.9 | 38.9 | 48.8 | 20.0/41.0 | 71.0 | 1.0 |

Wall time per step, seconds:

- olmo-3-7b: fetch_lenses 5, readout_filtered 31, readout_full 31, probe_swap 55
- olmo-3-7b-50pct: fetch_lenses 0, readout_filtered 28, readout_full 28, probe_swap 42
