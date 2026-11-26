# J++ post evals: omni J-lenses and Neuronpedia baselines

jpp_lens d7efc0d. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at

the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary

(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.

| model | lens | filtered_macro_r@1 | filtered_macro_r@10 | multihop_r@10 | multilingual_r@10 | typo_r@10 | association_r@10 | poetry_r@10 | full_macro_r@1 | full_macro_r@10 | swap_top1 | swap_top1_hits | swap_top5 | swap_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-omni-7b | logit | 10.2 | 18.9 | 45.8 | 25.2 | 23.4 | 0.0 | 0.0 | 9.8 | 18.4 | 3.9 | 2.0/51.0 | 5.1 | 0.5 |
| qwen2.5-omni-7b | omni7b_jlens_n250 | 15.3 | 35.4 | 66.7 | 41.3 | 67.5 | 1.2 | 0.0 | 11.2 | 23.8 | 35.3 | 18.0/51.0 | 69.2 | 2.0 |
| qwen2.5-omni-7b | omni7b_jlens_n500 | 15.4 | 35.5 | 66.7 | 40.8 | 68.8 | 1.2 | 0.0 | 11.2 | 23.8 | 39.2 | 20.0/51.0 | 69.2 | 2.0 |
| qwen2.5-omni-7b | omni7b_jpp_n64 | 27.5 | 48.2 | 75.0 | 60.2 | 94.8 | 8.8 | 2.3 | 25.9 | 44.2 | 35.3 | 18.0/51.0 | 66.7 | 1.0 |
| qwen2.5-omni-7b | omni7b_rlens_n64 | 19.1 | 39.0 | 79.2 | 34.0 | 81.8 | 0.0 | 0.0 | 14.0 | 29.0 | 29.4 | 15.0/51.0 | 71.8 | 2.0 |

Wall time per step, seconds:

- qwen2.5-omni-7b: fetch_lenses 2, readout_filtered 74, readout_full 70, probe_swap 168
