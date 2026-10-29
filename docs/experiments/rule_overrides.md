# Rules set aside or changed after a result

Four times a rule fixed before a run was set aside or changed after a result
had been seen, by an explicit decision recorded with its date and reason. Two
are overrides of a decision rule (1 and 2); two are changes to a criterion or
to a flag rule made after its outcome was known (3 and 4). All four are
declared in the paper.

## 1. The final rule of H3.1 (23 October 2026)

**The rule, as fixed.** The onset tracking would be adopted only if it passed
every binding criterion, on the development lines (1–8, 11) and on the
synthetics (9, 10), and then the held-out test on MS-29; if a single binding
criterion failed, the stage-1 tracking would be released and the development
of the picking closed, with no further version.

**The outcome.** On the development lines every binding criterion passed.
Criteria 9 and 10 failed for both variants: coverage of 86 % (h2) and 75 %
(h2m) against the 90 % required, and 23 / 21 (h2) and 24 / 19 (h2m) bench
lines of 36 against the 33 required (`onset_tracking.md`). By the rule, the
stage-1 tracking was to be released.

**The decision.** The rule was set aside, before any further run, and the
onset tracking (h2, complete method) went to the held-out test on MS-29 with
the criteria a–e already fixed. No new version was made: the method stayed
as frozen for the formal evaluation.

**The reason.** The rule compared the onset tracking with absolute thresholds
and kept the stage-1 tracking as the safe fallback, but on every comparable
measure the fallback was below the onset tracking, including on the criteria
the onset tracking had missed:

| measure | stage 1 | onset tracking |
|---|---|---|
| IQR against the manual picks, Viking / IT90-65 / SISOC-MEPA | 18.4 / 8.5 / 15.5 ms | 3.9 / 2.5 / 2.5 ms |
| R_wb discrepancy, IT90-65 / SISOC-MEPA | 117 / 59 % | 10 / 6 % |
| synthetic anchors on the right cycle, same records | 67 % | 97–98 % |
| bench C lines passing, R0 / R1 | 14 / 11 | 23 / 21 |

**The exception of Viking.** On one comparable measure the stage-1 tracking
is better: the R_wb discrepancy on Viking, 33 % against 59 % for the onset
tracking. "Dominated" holds only with Viking's R_wb excluded, for the reason
given in 3 below: at 4-ms sampling that discrepancy measures which cycle the
matched filter of the multiple locks on, not the tracking.

With that exclusion the fallback was dominated. The same defect had already
been recorded for the decision rule of the hybrid pickers (22 October). From
then on the question was whether the onset tracking was better than the stage-1 tracking, not
whether it reached absolute thresholds. The variant h2 was chosen for its
higher synthetic coverage, without reference to the real lines.

## 2. Criterion 1 of the autocorrelation test (26 October 2026)

**The rule, as fixed.** The autocorrelation test of the operating domain
(`operating_domain.md`) would be adopted only if (1) the four off-cycle
records at the edge of the domain were flagged outside, (2) at most 5 % of
the records well inside the domain were flagged outside by the new test, and
(3) no real line came out outside; if a criterion failed, the work would stop
and the detail be reported.

**The outcome.** Criterion 1 failed: two of the four records (tracking errors
of +429 and +273 ms) were flagged outside, two (+59 and +96 ms) were not.
Criterion 2 passed (1 of 202 records, 0.5 %), criterion 3 passed.

**The decision.** The test was adopted, declaring criterion 1 as not met,
together with the limit it showed: a seafloor within a few tens of
milliseconds of the end of a long direct arrival is not recognised by these
measures. In one of the two missed records the seafloor lies 3 ms inside the
end of the direct; in the other the autocorrelation locked on a later lobe.

**The reason.** The alternative, the flag without the autocorrelation test,
recognised none of the four cases; the test recognised the two gross errors
with 0.5 % false alarms, and changes neither tracking nor wavelets. A
dominated alternative was not kept, as in the first decision.

## 3. Criteria 2 and 3 on Viking made non-binding (23 October 2026)

**The rule, as fixed.** Binding criteria of the onset tracking, fixed on 22
October before any code: on Viking, (2) a discrepancy of the two R_wb
estimates of at most 43 % (the stage-1 value, 33 %, plus ten points) and (3) a
depth from the multiple not worse than the stage-1 value (2.4 %), with the
multiple not at the search boundary. On IT90-65 and SISOC-MEPA the thresholds
were < 25 % and ≤ 2 %.

**The outcome seen.** In the formal evaluation of H3 (22 October) both failed
on Viking: R_wb 59 % (h2m 57 %), depth +2.9 % (h2m +3.4 %).

**The decision.** In the protocol of H3.1, the next day, criteria 2 and 3 on
Viking became reported and not binding; on IT90-65 and SISOC-MEPA they stayed
binding with the same thresholds. The change was recorded as made after the
outcome of H3 had been seen; the outcome of H3.1 was not yet known.

**The reason.** At 4-ms sampling the matched filter of the first multiple
locks one cycle late on Viking in the stage-1 tracking as well: the R_wb of
the stage-1 tracking is consistent only because its multiple window sits one
cycle later. The two criteria measure that lock, not the tracking.

**What it changed.** In the formal evaluation of H3.1, Viking gave 59 % on
criterion 2, which would have failed as a binding criterion, and +0.3 % on
criterion 3, which would have passed. The outcome of H3.1 by the rule was the
same either way, since criteria 9 and 10 failed. The change matters through
the first override: the statement that the stage-1 fallback was dominated
holds only with Viking's R_wb excluded on this ground.

## 4. "Partially outside" decided from 100-shot blocks only (26 October 2026)

**The rule, as fixed.** The flag of the operating domain, approved on 23
October: "inside" when no shot is below the threshold, "outside" when more
than 50 % are, "partially outside" otherwise; one shot below was enough for
"partially outside".

**The outcome seen.** In the run of the autocorrelation test (26 October),
IT90-65, SISOC-MEPA and MS-29 became "partially outside" through isolated
shots, 1.3 %, 0.6 % and 0.1 % of the line, with no 100-shot block below the
threshold; Viking stayed inside. Criterion 3 of that test (no real line
"outside") passed either way.

**The decision.** "Partially outside" is decided from the 100-shot blocks
only: a block whose median separation is below the threshold, or more than
half of whose shots fail the autocorrelation test. Isolated shots are counted
and reported, and still count towards the 50 % of "outside". With this rule
the four lines are inside.

**The reason.** The record gives none beyond the proposal itself. The rule
was one of three options listed with the outcome: (a) start the
autocorrelation search at the end of the direct, (b) declare the limit, (c)
decide "partially outside" from the 100-shot blocks only. The decision took
(b) and (c) and rejected (a).
