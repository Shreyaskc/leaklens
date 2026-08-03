# leaklens — Benchmark Contamination Detection for LLMs

**Class:** Measurement/rigor tool (Python toolkit)
**Citation anchor:** arXiv paper (ACL/EMNLP submission) + JOSS
**Status:** Planning — this document is the implementation brief.

---

## Summary

`leaklens` is a unified toolkit that answers: *"has this model likely seen this
benchmark during training?"* It implements the major published contamination
detection methods behind one API — n-gram/substring overlap against open
corpora, perplexity/min-k% membership inference, guided completion tests
(can the model complete a benchmark item verbatim?), order-canary tests
(does the model know the benchmark's item ordering?), and performance-vs-
paraphrase gap probes — and produces a standardized **contamination report
card** for any (model, benchmark) pair. It also maintains a public registry of
report cards for popular model × benchmark combinations.

## The problem it solves

Benchmark contamination is the central validity threat in LLM evaluation:
scores on GSM8K/MMLU-era benchmarks are partly memorization. The detection
literature is rich (Carlini et al. membership inference; min-k% prob; guided
completion; "Data Contamination Quiz"-style probes) but **fragmented across
one-off repos with incompatible interfaces and no maintained implementation**.
Every eval paper that wants to address contamination — and reviewers now
routinely demand it — must reimplement a detector or wave hands. There is no
`pip install` answer.

## Why it matters

If contamination is unmeasured, benchmark results are uninterpretable, and the
field's headline claims rest on sand. A standard, maintained detector (a) lets
every eval paper include a contamination check with one command, (b) enables
meta-science: tracking which benchmarks have died, and (c) is a prerequisite
for certifying new benchmarks clean (this portfolio's `evergreen-qa` and
`indicreason` will ship leaklens report cards — built-in first citations).

## Citation thesis

- Reviewer-driven demand: "did you check for contamination?" needs a citable
  one-line answer, exactly the dynamic that made rigor tools highly cited.
- Aggregator position: papers citing leaklens will cite it *alongside* the
  original methods, not instead — so method authors have incentive to
  contribute rather than compete.
- The public report-card registry is independently citable ("GSM8K shows
  strong contamination signal for model X [cite]").

## Deliverables

1. `pip install leaklens`:
   - `leaklens.audit(model, benchmark)` → report card with per-method scores,
     calibrated against clean/contaminated reference pairs.
   - Detectors (each a plugin with a common interface, citing its source paper):
     `ngram_overlap` (vs Dolma/C4/FineWeb indexes via infini-gram API),
     `min_k_prob`, `perplexity_gap`, `guided_completion`, `order_canary`,
     `paraphrase_gap` (score on original vs meaning-preserving paraphrases).
   - Works with local HF models (logprob-based methods) and API models
     (completion/behavioral methods only — the report card marks which
     methods were applicable).
   - Benchmark adapters: any HF dataset + built-ins for MMLU, GSM8K,
     HumanEval, ARC, HellaSwag, TruthfulQA.
2. **Calibration suite**: a set of (deliberately-memorized, held-out) model
   pairs built by fine-tuning small open models with/without benchmark data —
   used to report each detector's true/false positive rates. This is the
   scientific contribution beyond aggregation.
3. **Registry**: JSON report cards for ~10 popular models × 8 benchmarks,
   published on GitHub Pages + HF.
4. Paper: toolkit + calibration study + registry findings ("which public
   benchmarks are dead?").

## Implementation plan

**Phase 1 — Framework + easy detectors (week 1–2).** Plugin interface;
benchmark adapters; `ngram_overlap` via the infini-gram public API (no local
corpus index needed); `guided_completion` and `order_canary` (work on API
models). CLI: `leaklens audit --model X --benchmark gsm8k`.

**Phase 2 — Logprob detectors (week 3).** `min_k_prob`, `perplexity_gap`,
`paraphrase_gap` (paraphrases generated once per benchmark, human-spot-checked,
shipped as data so results are reproducible).

**Phase 3 — Calibration (week 4–5).** Fine-tune 2 small open models (e.g.,
1B–3B) on contaminated vs clean mixes; run all detectors; report ROC curves.
This produces the paper's core table: detector sensitivity/specificity.

**Phase 4 — Registry + paper (week 6–7).** Run audits across the model ×
benchmark matrix (API + open models); publish registry; write paper with the
registry findings as the headline. arXiv → ACL/EMNLP; toolkit → JOSS.

## Validation

- Calibration ROC on known-contaminated fine-tunes (Phase 3).
- Reproduce at least one published contamination finding as a sanity anchor.
- Negative control: post-cutoff benchmark items must score clean.

## Release checklist

CITATION.cff → PyPI → Zenodo → arXiv → JOSS → registry site → HF Space
("paste a benchmark, pick a model, get a report card") → announcement thread
led by the "which benchmarks are dead" figure → seed emails to eval-paper
authors and benchmark maintainers → offer lm-eval-harness a `--contamination`
flag.
