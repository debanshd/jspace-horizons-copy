# J++ post evals: omni J-lenses and Neuronpedia baselines

jpp_lens d7efc0d. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at

the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary

(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.

| model | lens | filtered_macro_r@1 | filtered_macro_r@10 | multihop_r@10 | multilingual_r@10 | typo_r@10 | association_r@10 | poetry_r@10 | full_macro_r@1 | full_macro_r@10 | swap_top1 | swap_top1_hits | swap_top5 | swap_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-omni-7b | logit | 9.0 | 17.4 | 38.5 | 25.2 | 23.4 | 0.0 | 0.0 | 8.6 | 16.6 | 3.9 | 2.0/51.0 | 5.1 | 0.5 |
| qwen2.5-omni-7b | omni7b_jlens_n250 | 14.0 | 34.2 | 63.5 | 40.3 | 66.2 | 1.2 | 0.0 | 9.9 | 22.8 | 35.3 | 18.0/51.0 | 69.2 | 2.0 |
| qwen2.5-omni-7b | omni7b_jlens_n500 | 14.1 | 34.1 | 59.6 | 40.8 | 68.8 | 1.2 | 0.0 | 9.9 | 22.8 | 37.3 | 19.0/51.0 | 69.2 | 2.0 |
| qwen2.5-omni-7b | omni7b_jpp_n550 | 27.9 | 47.3 | 65.4 | 60.1 | 94.8 | 13.8 | 2.3 | 26.4 | 45.0 | 39.2 | 20.0/51.0 | 82.1 | 2.0 |
| qwen2.5-omni-7b | omni7b_rlens_n550 | 17.8 | 37.3 | 75.0 | 33.5 | 77.9 | 0.0 | 0.0 | 12.6 | 26.5 | 29.4 | 15.0/51.0 | 74.4 | 2.0 |

Wall time per step, seconds:

- qwen2.5-omni-7b: fetch_lenses 21, readout_filtered 45, readout_full 44, probe_swap 83
