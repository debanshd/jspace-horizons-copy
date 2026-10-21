# Examples

Worked examples from the J-space runs, for the report and for slides. Everything here was made from the runs' own logs by jbound/fig_examples.py; nothing was re-run.

ex1 to ex8 are the figures, each as .png, .pdf and .svg. The SVG keeps its text as text: PowerPoint opens it and "Convert to Shape" splits it into separate pieces, and Keynote, Figma and Illustrator open it with layers. Google Slides does not import SVG, so use the PNG there.

inputs/ has every input at full size, exactly as the model received it. The canvases are cut from the stimulus sheets the runs saved; the empty and printed-word canvases are drawn with the run's own code; the SCAM and GQA photos are resized with the run's own code. A file ending in _slotN.png is the one slot the figures show.

outputs.csv has one row per answer shown in the figures: the model, the input file, the exact prompt (the user turn), what was done inside the model, the answer as the model wrote it (output_verbatim), the answer as the figure prints it (output_shown), and the source CSV with the row. The rule questions asked for 7 (yes) or 3 (no); output_shown has yes and no.

To rebuild: venv/bin/python jbound/fig_examples.py. It needs data/coco2017, data/scam and data/bench/gqa, and the TeX Gyre Heros font (apt package fonts-texgyre).

## 21 October examples (ex9 to ex17)

Built by scripts/fig_examples21.py from results/oct21 with the same conventions: the canvases are rebuilt from each run's crops.json with the run's own stimulus code, every answer is read from the run's CSV, nothing is re-run, and outputs_21oct.csv has one row per answer shown (the source CSV and row beside it), in the columns of outputs.csv. Each cell carries, in small type, how many of the run's 64 scenes gave that cell's outcome, so the single scene and the population are read together. ex10 to ex12 show two scenes; ex15 and ex16 use the natural photos (GQA, CountBench) and need data/gqa and data/countbench to rebuild.

- ex9_word_read_into_picture: the reader heads' outputs recorded while the 2B reads "Image slot 1 contains a bird." are written into the run that sees an airplane, per head, per layer and jointly, and the reverse (RECORD.md row 101).
- ex10_saw_what_it_read: given only the name station's output of a run that read "slot 1 contains a bird", the 4B names the bird and says it saw it; with the middle layers added it says it read it (rows 103, 105).
- ex11_two_sources_in_conflict: a horse in the picture and a note that says cup; with the picture's reads blocked the 4B names the cup and says it read it (row 110).
- ex12_same_answer_different_cause: a horse in the picture and a note that says horse; the answer is horse under every block and the report follows the pathway that carried it (row 115).
- ex13_nothing_blocked: the heads of a run that answered from the note, written in with the picture present and unblocked, make the 4B say it read the answer; another horse's heads and matched noise do not (row 129).
- ex14_which_token_reads: on the 2B, keeping only the word "animal" from reading the picture flips the animal rule, keeping only the answer position from it does not (row 109).
- ex15_photo_blank_state: a GQA photo with its three questions; a grey image's token states written in from layer L on; the decision stops needing the photo by layer 11, the attribute by 15 to 23, the name by 19 (row 91).
- ex16_counting_reads_deeper: three CountBench photos; the question's reads of the image tokens blocked from layer L on; a count of 2 is back by layer 11, 6 by 19, 7 only from 27 (row 88).
- ex17_eleven_questions_one_scene: the question battery on one scene of the 4B (the 11 of 18 questions it answers right with nothing patched), patched one layer at a time; every copy of the word form follows the donor until layer 22 and every decision until 9 or 10 (row 80).

To rebuild: .venv/bin/python scripts/fig_examples21.py, with the TeX Gyre Heros font installed. With data/coco2017 present the canvases are cut from COCO; without it the script uses results/samples/<model>/bank (every tile of each model's crop bank, crop set 1, as PNG) and results/samples/<model>/scenes (every receiver canvas of the 64 scenes, with scenes.csv listing X, Z, the slot and the name set per scene), which the same script writes.
