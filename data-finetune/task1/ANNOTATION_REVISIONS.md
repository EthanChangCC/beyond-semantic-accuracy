# Task 1 test-set annotation revisions

This directory ships two revisions of the same 500-utterance Task 1 test set. They have
identical records, ids, order and sentences; they differ only in `tokens` (field spans) and
`utterance_action`.

| File | Use |
|---|---|
| `eval.jsonl` | **The annotations the conference paper's reported numbers were computed on.** Use this to reproduce the paper. |
| `eval_revised.jsonl` | A later, audited revision. Use this for new work. |

`eval_revised.jsonl` differs from `eval.jsonl` on **113 of 500 samples** (110 with changed
field spans, 17 with changed action sets). It is the result of three audit rounds:

1. **Consistency unification (92 samples).** The set contains 19 groups of repeated
   transmissions; in 10 of them the repeated instances had carried conflicting labels. A
   further class of samples had control values absorbed into action spans, which left the value
   unscored. This round was performed blind — without reference to any model output — and
   unified these conventions and restored the missing values.
2. **Span-coverage repair (11 samples).** Remaining cases where the token sequence did not
   cover the whole utterance.
3. **Adjudication of unanimous failures (28 samples).** Every sample that *all* evaluated
   systems answered incorrectly (48 samples) was re-examined by hand; 28 were found to carry an
   annotation error and were corrected, 19 were confirmed correct, and 1 was left unchanged as
   undecided. Because this round selects samples by model behaviour, it is listed separately
   from the blind rounds above.

## Effect on results

Revision raises the measured score of most systems, because previously unscored critical fields
become scored and because the hardest samples included annotation errors. Rankings are
unaffected in our experiments. Round 3 raises the strongest single-model baseline by roughly
four points, so a comparison must never mix the two files.

**Report which file was used.** Numbers computed on `eval_revised.jsonl` are not comparable to
the numbers printed in the conference paper.
