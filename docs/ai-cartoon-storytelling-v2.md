# AI Cartoon Storytelling V2 — local execution record

All work remains on `feature/ai-cartoon-storytelling-v2`. Baseline:
`0a0c8c1230c293776c20f54da78043419d5c0865`. No merge to main.

Execute phases 1–7 sequentially. Each checkpoint requires compile, Ruff,
focused tests, full regression suite, relevant real media inspection, local
commit and push. Ordinary defects are repaired before advancing.

## Phase 1 — semantic narration beats (verified)

The script now determines beat boundaries. Small EN/ES clause rules distinguish
contrast, analogy and consequence; introductory conditions stay with the action.
SRT token matching supplies monotone anchors, and internal positions use explicit
proportional estimates. No duration threshold merges independent sentences.

`timing_source` records the start-boundary origin: `direct_srt`,
`interpolated_srt`, or `fallback_proportional`. Direct SRT is not a claim of
phoneme measurement: the baseline Chatterbox subtitles themselves are estimated.
The timeline spans the real audio duration, including reused preview audio;
other video sources retain their existing rounded-duration behavior.

Reference source (preserved):
`storage/tasks/721f5d9e-e19d-489a-b1ac-a03322e07058`.
Audio duration: 21.340 s; former plan duration: 22 s.

Reproduction before implementation: 11 failed, 1 passed in the new timeline
tests. The email and instruction were incorrectly merged.

Focused gate after implementation: 108 passed, 2 skipped, 18 subtests passed.
Compile: exit 0. Ruff: all checks passed.
Full suite after fixture portability fixes: 1087 passed, 11 skipped,
8558 subtests passed in 259.54 seconds (15 warnings). Existing integration skips
remain; no new skip/xfail was introduced.

Real render: `storage/tasks/cartoon-v2-phase1-reference-1dbb9282`.
All four required artifacts exist. Both MP4s fully decoded with FFmpeg exit 0.
Material: 1080x1920, 24 fps, 21.375 s. Final: 1080x1920, existing compositor
30 fps, 21.366667 s video / 21.340 s audio. Frame rounding is below one frame.
Eight contiguous beats: instruction begins at 14.244 s, payoff at 19.451 s;
last beat ends at 21.340 s. Boundary contact sheets inspect both material and
subtitled final at 14.15/14.30, 19.35/19.55 and 21.30 seconds. No black tail.

Remaining visual limitations belong to subsequent phases: generic/static cards,
clipped long card labels, alternating narrator ownership and incomplete visual
payoff. Phase 1 certifies segmentation/coverage, not storytelling quality.
The first console render used Windows cp1252 and emitted logging encoding errors;
the pipeline still succeeded. Use Python `-X utf8` for subsequent render logs.

First full gate: 2 failed, 1085 passed, 11 skipped, 8558 subtests passed.
Two unrelated validation issues were isolated: the corrupted-cache test could
take the future-mtime branch due to Windows clock precision, and the headless
folder test mocked Linux detection but expected Linux path separators on Windows.
The cache fixture now supplies an explicit fresh timestamp; the folder assertion
uses native separators. Assertions about warnings, cache removal and the exact
mapped task path remain intact. Their focused rerun plus timeline: 42 passed.

Reference verification runs the real `task.start()` pipeline with its existing
voice-preview input, supplying preserved narration/cues in a new task directory:

```powershell
uv run --no-sync python -X utf8 tools/verify_cartoon_storytelling.py --phase 1 --mode reference --reference-dir storage/tasks/721f5d9e-e19d-489a-b1ac-a03322e07058
```

## Remaining sequence

2. Bounded storytelling direction with validated actions and states.
3. Persistent objects and visible cause/effect; parked is not completed.
4. Host narration independent of visual actor and guest reactions.
5. Deterministic within-beat action and resolution.
6. Staging and subtitle safe zones, cartoon-only adaptation.
7. Audio-aware mouth smoothing and final acting; fresh Chatterbox/Ollama E2E.

Final acceptance also requires a second semantic scenario, deterministic
fallback, full timeline/compositor preservation and muted/no-subtitle review.
