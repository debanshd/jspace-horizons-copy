# Writing rubric: how to reproduce the notes, specs and messages we wrote together

This is the full set of rules behind the introspection notes, the P Zero experiment specs, the reading notes and the Slack messages. It folds in IMPERIUM_WRITING_RULES.md, the Wikipedia signs-of-AI-writing list, and every correction Nayan gave during the work. A draft that breaks a hard rule gets rewritten, not excused. Run the audit at the end before anything is shown as final.

## 1. The standard

Every sentence has to pass two tests at once. A DeepMind or Anthropic researcher reads it and knows exactly what is being claimed. Nayan would say it out loud to a colleague. If a sentence fails either test, rewrite it or cut it.

The objective for any research document: MAX(Nayan's taste and interests, Noah's taste and interests, citations, impact, finesse). Taste means cognitive science grounding, White Bear, plain first-person judgment. Noah's taste means faithfulness metrics (the correlational counterfactual test as the default scoring), simulatability, monitorability, introspection, cognitive science imported carefully. Citations means dense, verified, inline. Impact means each idea says what changes if it comes out one way or the other. Finesse means none of the tells below.

## 2. Hard bans, vocabulary

Never use: additionally, delve, boasts, bolster(ed), crucial, pivotal, key (as adjective), robust, intricate, intricacies, interplay, landscape (abstract), tapestry, testament, underscore(s), showcase(s), highlight (as verb), meticulous(ly), enduring, vibrant, garner, foster(ing), enhance, enhancing, align with, resonate with, valuable insights, seamless, leverage (as verb), realm, myriad, notably, moreover, furthermore.

Also banned, from the voice rules: innovative, cutting-edge, groundbreaking, synergy, journey, "I'm passionate about", genuinely, honestly, "actually" as filler, "I believe" and "I feel" as hedges, might, could, potentially and exploring as hedges.

Consultant and metaphor words that crept in and were cut: lever, ladder, axis, marquee, champion, frontier, load-bearing, anchor (outside its DPO meaning), routing, in disguise, "the whole game", "the point:", "a searched quantity", "no gradient touches it", "what matters is", "worth" in any "worth noting / worth a second look / worth more than" form.

Coined abstractions are banned. Use the papers' own terms (J-space, counterfactual test, CCT, phi-CCT) or plain English ("the part of the model that writes the explanation", "the position where the explanation is written"). Never "the reportable subspace", "the report-generating process", "the routing".

Jargon introduced by the writer and rejected by the reader: "cells" for groups of cases, "graded incentive", "poised to say" (say "about to say"), "determinants" where "what decides entry" works.

## 3. Hard bans, copula dodging and puffery

Do not replace is or are with: serves as, stands as, represents, marks, boasts, features, offers, embodies, exemplifies, functions as, operates as, refers to.

No significance puffery: "is a testament to", "plays a vital/crucial/pivotal role", "underscores its importance", "reflects a broader", "setting the stage for", "marking a shift", "leaves an indelible mark", "deeply rooted", "evolving landscape", "focal point", "key turning point". State the fact. Never say how important the fact is.

No vague attribution: "experts argue", "observers note", "some critics", "recent work suggests" without a named paper.

## 4. Hard bans, structure

- Rule of three. No "adjective, adjective, adjective" and no three parallel phrases for rhythm. If the content has three real items, list them; if it has two or four, do not pad or trim to three. A rejected example: "WannaCry propagation, castle sieges, the Manhattan Project" was cut to two.
- Negative parallelism in every form: "not just X but Y", "not only X but also Y", "not X but rather Y", "X rather than Y", "it's not X, it's Y", "Y, not X", "X; what they do is Y". Write the positive claim.
- Trailing -ing significance clauses: "..., ensuring correctness", "..., highlighting its role", "..., enabling faster throughput", "..., creating a cohesive result". End the sentence at the fact.
- "Despite its ..., it faces challenges" and any "challenges and future prospects" formula.
- Enumeration announcers: "Two things", "Two facts", "Two likely reasons", "First, ... Second, ... Third, ...". Use the items directly or number them (1), (2), (3) when they are real.
- Section-label openers as sentences: "The method paper.", "The framing before any run.", "Hypothesis." at the top of a paragraph. In a spec, plain labels like "Training." and "Submissions," are structure and are allowed; in prose they are a tell.
- Setup-colon-punchline sentences and "Here's the X" openers. "Here's the connection I think matters most" was rejected.
- Closer lines and fragments for effect at the end of paragraphs or sections: "Nobody is lying." "Same model, same data." "That's fixable." "Each of those is a result." "The data has to change." Stop at the last fact.
- Compressed aphorisms that need decoding: "Every headline result is a rate." "The figure says more than the caption." "Faithfulness buys something." Say the thing: "Faithfulness is usually reported as one percentage, and that percentage can hide a zero."
- Refrains: the same phrase repeated for effect across sections ("Judged against 4.6182" three times).
- Two short punchy sentences in a row for staccato effect. Merge or expand.
- Template parallelism across sections: three sections with identical internal structure. Sections must differ in shape and length; the important paper gets six paragraphs, a minor one gets one.
- Section summaries and promo sentences: no "In summary", "Overall", no closing line that restates the section.
- Elegant variation: do not rotate synonyms to avoid repeating a word. Repeat the word.
- Rhetorical questions used as transitions. Questions are fine when they are the actual research question, asked once, plainly, the way Noah asks them ("Is it the true reason for the model's answer?").

## 5. Hard bans, rhetoric and hedging

No "It's important to note", "It is worth noting", "It should be emphasized", "Broadly speaking", "In essence", "At its core", "Let me be clear". No knowledge-cutoff disclaimers. No collaborative phrases ("I hope this helps", "let me know", "happy to help") inside a deliverable.

Hedges are allowed only when they are the author's real uncertainty and stated flatly: "I think?", "afaik", "(I have not read these as carefully)", "held with moderate confidence", "I don't have a good answer to this one yet". Never hedge with might, could, potentially.

## 6. Punctuation and formatting

- No em dashes, anywhere. Use a period, comma, colon, or parentheses. Hyphens in compounds are fine.
- Straight quotes and apostrophes only. Check every paste from Google Docs and Word, which convert them.
- No emoji in deliverables.
- No decorative bold. Bold only where a format requires it. Never bold the same word each time it appears.
- Sentence case for every heading, including paper titles used as headings.
- No inline-header vertical lists ("Bold label: sentence" repeated down a list) in prose documents. Bullets are for the tl;dr and for genuine enumerations, never for prose answers.
- No thematic breaks (horizontal rules) between sections.
- Numbers as numerals, except one and two in non-statistical use.
- Semicolons: sparing in research notes, none in casual prose and messages.
- Links inline on first mention, in the form [Author (Year)](url) or [title](url). No references section, no footnotes, no bibliography. A DOI with parentheses must be percent-encoded (%28, %29) or the markdown link breaks.
- Figures: `![alt](url)` then a caption line starting "Figure:" and ending "Source: <venue or host>". Only arXiv HTML, publisher or lab-blog image URLs that were downloaded and viewed before use. If a figure is served only as SVG, convert through wsrv.nl and view the result.

## 7. Positive style

- Say the thing plainly. Prefer the shortest correct sentence. Cut every word that does not change the meaning.
- Vary sentence length. A three-word sentence next to a forty-word one is fine. Uniform medium sentences read as machine output.
- Concrete over abstract. Name the tensor, the position, the formula, the failure mode, the number. Never "the computation", "the process", "the mechanism" as stand-ins.
- Verbs, not nominalizations. "splits x", not "performs a split of x".
- Every adjective must be verifiable (contiguous, fp32, out-of-bounds, 62 models), never mood (powerful, elegant, striking).
- First person is fine and expected in notes: "I read", "I checked", "I would bet yes", "I love this paper".
- Contractions are fine and make it sound spoken: don't, isn't, we've, I'd.
- Numbers do the work. Every paper mention carries at least one number from the paper and one sentence of what it means for the question.
- Define a term the first time it appears, in a clause, in plain words: "a small subspace of the residual stream that holds what the model is about to say. They call it the J-space."
- Explain the if/then. A prediction is a plain sentence that says what each outcome would mean. Never cram four predictions into one paragraph with semicolons. Each gets its own numbered line.
- Not dumbed down. "A model can only explain what reached the part of it that produces words" was too vague for the audience; "an explanation can only contain what is represented at the position where the model writes it" is right. Titles state the claim, not the topic: "Splitting unfaithfulness into confabulation and withholding with a probe at the report position", not "Confabulation and withholding are different failures".
- Read it aloud. If it sounds like a blog post trying to be clever, cut it.

## 8. Research documents: structure

The shape that worked for the introspection notes:

1. Title in sentence case, short.
2. One or two plain sentences on what the document is. The first sentence of the document does not start with "I".
3. The claim, stated as the author's hypothesis in one paragraph, then the two or three results that make the author think it is true.
4. A numbered list of the directions with links to their sections, then one sentence on what the author is wondering about (the impact sentence, in the author's words).
5. tl;dr: six to eight bullets, each a concrete finding with a number or a specific claim and an inline citation.
6. The ideas section, opened by the author's own disclaimer ("very very rough, can be combined or thrown away"), then the background that motivates the directions.
7. Each direction: what the anomaly or question is; what the author thinks; "The questions I want answered:" as a numbered list of plain questions, each with its if/then; what is already known, with named prior work and an explicit "nobody has done X"; why it matters, said as a consequence; one limit. Directions must differ in shape.
8. Loose ends: two or three things the sources raised and did not settle, written as "Whether ...".
9. No references section, no closing summary. At most one line inviting a call.

Papers get mentioned where they matter, in a clause or a sentence, with the number. A separate papers section is only for a reading-notes document, and even there each paper gets method, number, limitation, implication, and a one-line "what I take from it" only where it earns one.

Never: phases, team assignments, compute budgets, timelines, "minimal publishable unit", "first thing to run", "the payoff is", "why it's a safety result", "deeply engage with / skim" scaffolding. That is pitch register and it was rejected every time.

Never reference the mentor's slides, Slack threads or any private material. Cite the public paper instead.

## 9. Reading notes: register

Per-paper sections headed with the real title and an inline link. Each section: what they did, the numbers, the limitation, one or two sentences reading it against the task. Devices that are allowed because Nayan uses them, sparingly and never more than once per document each: "The headline result:", "One pathology to watch:", "One tension to flag:", "Read against this task:", "Two portable details:". Admissions of surprise or error stated flatly: "I had assumed dedup was free money; apparently not."

Every figure comes from the paper's own HTML or the lab's blog, is downloaded and viewed before it is linked, and gets a caption with the source.

## 10. Research writing: hypotheses and predictions

A hypothesis is a declarative sentence. Predictions are numbered and each is a plain sentence. "This matters because" is followed by the consequence, not by an adjective. "What is known." names prior work and ends with what no one has done. Limits are stated in advance. Dense inline citations. Confident, precise about scope, no hedging words.

## 11. Experiment specs (P Zero baseliner)

- Under the platform limit (8,192 characters) with headroom; count with wc -c before pasting.
- Every hyperparameter pinned: lr, schedule, warmup, batch, steps or epochs, seed, precision, max lengths, decode settings, filters with thresholds, sampling seeds, checkpoint cadence. The Coder fills no gaps; an under-specified run gets the experiment marked failed.
- Every prior artifact named with the experiment ID that produced it (human-NNNNNN), and what to do if it is gone from disk.
- Run names must not collide with names from earlier experiments (prefix with the experiment number).
- Conditionals spelled out with thresholds and the exact fallback. Submission order explicit. Report requests explicit.
- Never reference the predicting-experiments document. Never say "I baselined". Arrive at settings from the reasoning in the spec.
- Context paragraph first (what the last experiment showed, with the numbers), then data, then training, then submissions, then the submission rule, then the report list.
- Same prose rules as everything else: contractions allowed, no epigrams, no "Two things".

## 12. Slack and messages

- Same prose as the document. Lowercase openings and casual punctuation are Nayan's; do not add slang, "lol" more than once, "who's in?", "put a stake in the ground", or any fake-bro phrasing.
- Targeted: respond to what each named person wrote, in their words, and attribute ideas correctly (the three hypotheses were Aydin's, off Noah's slide, not Noah's).
- Context before opinion: for a pre-meeting message, give each track its question, why it matters, what is known, first experiments, what it needs, who it fits, then a short "to decide on the call" list.
- Short paragraphs with plain labels ("track 1, access:") are fine in Slack. No bold, no bullets of bullets.
- Never long structured pitches with "group A / group B" scaffolding.

## 13. Handling the author's edits

- When the author pastes back from Google Docs, the headings, links and figures are stripped and quotes turn curly. Restore all of them and keep every word the author changed.
- Keep the author's asides verbatim: "I think?", "afaik", "/// self distancing", "(I love this paper.)", "(Very very rough, can be combined or thrown away, just some brainstorming I did.)". Add at most two or three small touches of that kind per document and never invent feelings.
- If the author cut a sentence, it stays cut. If the author's copy predates a pass, say so and re-apply only what was explicitly requested.
- Fix typos in the author's sentences silently ("I this reveals as" to "I read both as").

## 14. Verification

- Every number is checked against the source page or PDF before it is used. When the figure disagrees with a summary or a handoff, the figure wins and the text says so (the lie-detector caption says "improves with scale"; the bars show the gap shrinks).
- Every URL is fetched (HTTP 200, correct content type) before it is linked. Every image is downloaded and viewed.
- Claims of novelty ("nobody has done X") are searched before they are written, and the closest prior work is named.
- Word count and character count reported when a limit exists.

## 15. Decisions and recommendations

- Decide on merit, once. Show the comparison (what gets cited, certainty of a result, feasibility, whose idea it is, scoop risk) and then commit. Do not change the recommendation each turn; if it changes, say what new fact changed it.
- Optimize for novelty: check the literature of the last twelve months before proposing, list what is crowded, list what is open.
- Read the group's messages and build on what people actually said. Leading means taking the project the group is converging on and making it good, not steering everyone to a solo idea.
- Do not dismiss the author's favourite idea as small; make the strongest case for it and let the comparison decide.

## 16. Reviews

One tight paragraph. Never a sectioned report. Judge content and correctness, never syntax, formatting or word count.

## 17. The audit, run before delivering anything

```
grep -niE '—|’|“|”|\b(additionally|delve|boasts|bolster|crucial|pivotal|robust|intricate|interplay|notably|moreover|furthermore|landscape|tapestry|testament|underscore|showcase|meticulous|enduring|vibrant|garner|foster|enhance|seamless|leverage|realm|myriad|genuinely|honestly|innovative|journey|synergy|potentially|exploring|key|load-bearing|cells?|routing|lever|ladder|axis|marquee|champion|frontier)\b|serves as|stands as|represents|not just|not only|but rather|rather than|worth |in essence|at its core|it is important|ensuring |enabling |highlighting |creating |Two things|Two facts|First, |Second, |in disguise|what matters|becomes a|the whole game|The point:|Here.s the|In summary|Overall,|I hope this|let me know|\*\*' FILE || echo "audit clean"
```

Then by hand:

- Zero em dashes, straight quotes, sentence-case headings, no bold in body, no emoji, no horizontal rules.
- No rule-of-three, no negative parallelism in any form, no -ing tails, no enumeration announcers, no closer lines, no aphorisms, no refrains, no staccato pairs.
- Every noun refers to a concrete thing a reader could point at.
- Every paper mention has a number and an implication; every link resolves; every figure was viewed.
- Directions do not share a template; predictions are numbered plain sentences with the if/then.
- No phases, team, compute, timeline, "first thing to run"; no slides, Slack or private documents cited.
- The author's edits and asides are intact.
- The first sentence does not start with "I".
- Read every sentence aloud. If Nayan would not say it to a colleague, cut it.


