# Figures for the 22 October slides, in story order

All in jspace-horizons/runs/figures/ (PNG, PDF and SVG), each drawn from one results table by scripts/fig_battery.py. One question per figure. Numbers in RECORD.md rows 77 to 113.

1. figCircuit.png: the whiteboard. Two inputs (a picture, a sentence), three reads (what it is; the word form; which channel), three kinds of answer.
2. figStationBars.png: six models, two families, their layers as cells, the last layer that reads the picture for a decision (orange) and for the name (blue): 0.8B, 2B, 4B, 27B, Gemma E4B, Gemma 31B (the 9B joins when its knockout lands). The order holds everywhere; at 27B and 31B the reads are spread over many layers (figMatrix_Qwen3.5-27B.png, figMatrix_gemma-4-31B-it.png: no single layer costs a nat) and keep their shares of depth (31B: 0.48 and 0.70).
3. figR1_depth_Qwen3.5-4B.png: the single-layer patching sweep in nats, five questions, and every question's read depth; decisions early, copies late.
4. figMatrix_Qwen3.5-2B.png: the readers matrix, every question at every softmax layer (and figMatrix_gemma-4-E4B-it.png, every layer on Gemma).
5. figHeads_Qwen3.5-2B.png: the heads: 1 and 3 at layer 7 for decisions, 5 at 19 for the name, 4 at 11 shared.
6. figQuery_Qwen3.5-2B.png: which question token reads the picture: the word "animal" in the rule for the decision, the answer position for the name.
7. figCrossPositions_Qwen3.5-2B.png: where each head deposits the object it read, by question token.
8. figCardCross_Qwen3.5-2B_scene0.png: one scene, the interchange: the heads that read "bird" written into the model that sees an airplane.
9. figCrossModal_Qwen3.5-2B.png: the interchange over 64 scenes, per head, per layer, jointly, word donor against picture donor (and figCrossModal_Qwen3.5-4B.png; figCrossModal_gemma-4-E4B-it.png for the second family, where layer 29's heads carry the decision on 1.00).
10. figCrossAlign_Qwen3.5-2B.png: the vector a head writes for a dog seen against a dog read.
11. figCardSource_Qwen3.5-4B_scene0.png and figSource_Qwen3.5.png: given only the name station's output the model says it saw what it read; with the middle layers, that it read it.
12. figCardConflictSame_Qwen3.5-4B_scene0.png and figConflictSame_Qwen3.5-4B.png: the same answer from two sources; the 4B says "read" when the picture's reads are blocked and "seen" when the note's are, with the answer identical (64 of 64 and 62 of 62). The lead result for the source-report part. Then figCardConflict_Qwen3.5-4B_scene0.png and figConflict_Qwen3.5.png: two sources in conflict; the report follows the channel that caused the answer (53 of 53 flips).
13. figCount_Qwen3.5-4B.png and figGQA_Qwen3.5-4B.png: natural photos; counting reads deeper the more there is to count; GQA shows the same two stations.
14. figStations.png: the stations as layers and as shares of depth across models.

Limits to say on the slides: two Qwen sizes at the head level (the 4B's name does not transfer by head outputs); the 2B's source report is a format inference; human labels on the judge sheets still to be given; the 27B, 9B and 31B knockouts are running and will be added to figure 2.
15. figSourceCause_Qwen3.5-4B.png: the report follows which pathway's heads were written in, nothing blocked; noise of the same size and another picture of the same object move nothing (row 129).
16. figConflictSame_gemma-4-E4B-it.png: the answer-held source test in the second family (row 132).
17. figAccess.png: how the picture's read is spread over depth in all seven models; the position and order hold, the concentration on single layers falls with size (post hoc, figAccess.csv).
18. figMatrix_Qwen3.5-9B.png and the redrawn figStationBars.png / figStations.png with the 9B (row 131).
Boundary to say on the slide with 15: the pathway-tracking report exists only when the name is written before the channel (row 130).
