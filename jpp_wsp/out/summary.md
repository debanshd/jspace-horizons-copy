# J++ post evals: omni J-lenses and Neuronpedia baselines

jpp_lens d7efc0d. Recall in % of items (item weighting), best rank over the seven layers; probe-swap at

the scale with the best top-1 rate. `filtered`: Readout Filtering on (the post's J++ scoring); `full`: the whole vocabulary

(the post's J-lens and R-lens scoring). The post, Qwen3.6-27B: J-lens 35.7, R-lens 37.7, J++ 55.2 recall@10.

| model | lens | filtered_macro_r@1 | filtered_macro_r@10 | multihop_r@10 | multilingual_r@10 | typo_r@10 | association_r@10 | poetry_r@10 | full_macro_r@1 | full_macro_r@10 | swap_top1 | swap_top1_hits | swap_top5 | swap_scale |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| olmo-3-7b-50pct | logit | 8.9 | 28.1 | 70.8 | 33.3 | 24.7 | 6.5 | 5.3 | 8.6 | 28.1 | 9.8 | 4.0/41.0 | 54.8 | 2.0 |
| olmo-3-7b-50pct | olmo7b50pct_jpp_n64 | 21.2 | 39.5 | 58.3 | 26.3 | 83.1 | 24.7 | 5.3 | 18.9 | 38.9 | 48.8 | 20.0/41.0 | 71.0 | 1.0 |
| olmo-3-7b-50pct | wspP1_jpp_n64 | 24.4 | 40.8 | 66.7 | 26.9 | 87.0 | 23.4 | 0.0 | 22.9 | 39.8 | 39.0 | 16.0/41.0 | 71.0 | 1.0 |
| olmo-3-7b-50pct | wspP2_jpp_n64 | 21.5 | 42.3 | 58.3 | 30.8 | 87.0 | 24.7 | 10.5 | 18.6 | 40.6 | 46.3 | 19.0/41.0 | 71.0 | 1.0 |
| olmo-3-7b-50pct | wspP3_jpp_n64 | 17.3 | 34.7 | 50.0 | 31.4 | 76.6 | 15.6 | 0.0 | 16.8 | 34.2 | 48.8 | 20.0/41.0 | 67.7 | 1.0 |
| olmo-3-7b-50pct | wspV0_jpp_n64 | 17.2 | 36.1 | 58.3 | 31.4 | 75.3 | 15.6 | 0.0 | 16.4 | 34.7 | 51.2 | 21.0/41.0 | 67.7 | 1.0 |
| olmo-3-7b-50pct | wspV1_jpp_n64 | 26.6 | 46.2 | 75.0 | 37.8 | 85.7 | 27.3 | 5.3 | 23.1 | 43.5 | 39.0 | 16.0/41.0 | 77.4 | 2.0 |
| olmo-3-7b-50pct | wspV2_jpp_n64 | 22.5 | 49.1 | 75.0 | 45.5 | 88.3 | 31.2 | 5.3 | 19.4 | 47.0 | 43.9 | 18.0/41.0 | 67.7 | 1.0 |
| olmo-3-7b-50pct | wspV3_jpp_n64 | 17.4 | 35.5 | 58.3 | 33.3 | 68.8 | 16.9 | 0.0 | 16.9 | 34.7 | 48.8 | 20.0/41.0 | 67.7 | 1.0 |

Wall time per step, seconds:

- olmo-3-7b-50pct: fetch_lenses 0, readout_filtered 157, readout_full 165, probe_swap 463
