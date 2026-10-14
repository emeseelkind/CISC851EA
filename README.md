# prototypeProstEA

**Claim.** How core-level biopsy labels get turned into a training signal for micro-US cancer
detection is a design space, not a fixed choice. Multi-objective evolutionary search over that
space finds weak-supervision objectives that detect csPCa better and produce more spatially
consistent heatmaps than hand-designed losses -- without any pixel-level histopathology.

## Contributions

1. **Region-wise weak supervision, formulated as one parameterized objective.** Needle and
   prostate masks split each micro-US image into three regions: known-negative (outside the
   prostate), weakly labeled (the needle trace, via Gleason grade group and involvement), and
   unlabeled (prostate tissue outside the needle). This unifies MIL pooling, positive-unlabeled
   learning, and patient-level soft negatives into one design space. ProstNFound+'s own loss is a
   single fixed point in it (`prostea/genome.py::PROSTNFOUND_PLUS`).
2. **Multi-objective evolutionary loss search.** NSGA-II (`prostea/search.py`) searches pooling
   operator, GG target encoding, region weights, and the PU class prior. It jointly optimizes
   core-level csPCa AUROC and a localization-consistency proxy under an out-of-prostate activation
   constraint. Caching frozen foundation-model embeddings first (`scripts/cache_regions.py`) makes
   each genome's fitness a few epochs of linear-probe training, not a ViT forward+backward pass --
   tractable on a single GPU, or even CPU.
3. **Controlled empirical comparison.** Nested, patient-level cross-validation
   (`prostea/splits.py`) compares the discovered objective against ProstNFound+'s original loss,
   standard top-k MIL, and random search at equal evaluation budget (`scripts/run_search.py`,
   `scripts/random_search_baseline.py`).
4. **Design insight.** The Pareto front exposes the classification/localization-consistency
   trade-off that a single summed loss hides, and the search reveals which supervision choices
   actually move performance.

## Hypotheses

- **H1** -- the best evolved objective beats ProstNFound+'s loss on held-out csPCa AUROC by more
  than fold-to-fold variance. That variance is not hypothetical here: a real mean-pooled baseline
  on this exact cohort (`../prototypeProstate/experiments/baseline_mean_pool.json`) has csPCa AUROC
  0.678 +/- 0.030 across 5 patient-grouped folds (n=1118 cores, 76 patients). Gains under
  ~3 points won't clear that bar.
- **H2** -- NSGA-II beats random search at equal budget (`scripts/random_search_baseline.py`
  samples the same number of genomes uniformly from the same space).
- **H3** -- the out-of-prostate and PU/unlabeled-region terms improve the localization-consistency
  proxy without hurting AUROC (ablate by zeroing `w_negative` / `w_unlabeled`).

## What this project cannot claim

- **Localization accuracy.** There is no pixel-level histopathology ground truth in this data.
  `prostea/metrics.py::localization_consistency` measures whether the fraction of the needle trace
  a heatmap activates tracks involvement across cores -- a *consistency* proxy, never "accuracy."
- **Superiority to PRI-MUS or clinical readiness.** `pri_mus` scores (see Data below) are a
  reference point, not a target to beat.
- **Cross-site or cross-time generalization.** The only cached, needle-annotated cohort on this
  cluster with real prostate masks is single-center (UA), so nothing here is evidence of
  generalization across centers or cohorts unless a second center's data is added to the cache.

## Data (what's real vs. what the proposal assumed)

Everything the proposal assumed turns out to already exist as real data on this cluster, at
`/datasets/exactvu_pca/`:

- `OPTIMUM/UA_annotated_needles/<case>/<core>/` -- 1118 cores, per-core `image.png`,
  `needle_mask.png`, `micro_seg_net_prostate_mask.png` (**prostate masks exist**), `info.json` with
  `GG`, `% Cancer` (involvement), `PRI-MUS`, `center`, `case` (patient id).
  Read by `../prototypeProstate/protoprostate/optimum_data.py`.
- `checkpoint_store/pnf_plus_final.pth` -- a real trained ProstNFound+ checkpoint; its frozen image
  encoder loads via `../prototypeProstate/protoprostate/pnf_plus.py::load_pnf_plus`.
- `optimum_ua_pu_mil/` -- an existing fold split (`fold{0..4}_{train,val}.csv`) for
  positive-unlabeled MIL on this same cohort; worth cross-checking against
  `prostea/splits.py`'s own patient-grouped folds before trusting either.
- This cohort is **single center (UA)**. `../prostNfound/prostNfound/dataset.py`'s
  `CohortSelector.get_patient_splits_by_center` (leave-one-center-out) is code-ready but has no
  second center to run on yet -- see "What this project cannot claim" above.

## Reused from sibling projects (not reimplemented)

Both are siblings under `/project/6106383/emese/`; `prostea/_paths.py` adds them to `sys.path` so
`import protoprostate.*` works without installing anything.

| module | from | what |
|---|---|---|
| `protoprostate.pnf_plus` | `../prototypeProstate` | frozen PNF+ encoder load + forward |
| `protoprostate.optimum_data` | `../prototypeProstate` | raw OPTIMUM cohort walker |
| `protoprostate.labels` | `../prototypeProstate` | grade-string -> grade-group parsing |
| `protoprostate.splits` | `../prototypeProstate` | `patient_grouped_folds` (wrapped with nesting in `prostea/splits.py`) |
| `protoprostate.metrics` | `../prototypeProstate` | `core_metrics` (incl. `auc_cspca_max`, H1's metric), `per_center_metrics` |
| `CancerDetectionValidRegionLoss` pattern | `../prostNfound/prostNfound/losses.py` | the region + broadcast-label shape that `prostea/loss.py` generalizes; `../prostNfound/prostNfound/dataset.py`'s `CohortSelector` for if/when center-holdout data arrives |

`prostea/genome.py::PROSTNFOUND_PLUS` and `::TOPK_MIL` encode those two files' actual, shipped
defaults as fixed points in the search space -- see the docstrings there for exactly which config
values they're read off of (`../prostNfound/conf/main_training.json`,
`simple_mil_loss`'s `top_percentile=0.2` default).

## Layout

```
prostea/
  pooling.py       mean / max / top-k-mean / generalized-mean / LSE region pooling
  genome.py         the search space (dataclass) + PROSTNFOUND_PLUS / TOPK_MIL fixed points
  region_cache.py   loads scripts/cache_regions.py's full-patch-grid + mask cache
  model.py          LinearProbe -- the only trainable part (frozen embeddings otherwise)
  loss.py           genome -> trainable 3-region loss (weak BCE + negative BCE + nnPU)
  metrics.py        protoprostate.metrics re-exports + localization_consistency proxy
  splits.py         nested (outer test / inner search) patient-grouped CV
  fitness.py        train-and-eval one genome on cached embeddings
  search.py         pymoo NSGA-II problem + Pareto-front extraction
scripts/
  cache_regions.py(.sh)          build the region cache (GPU node, run once)
  run_search.py(.sh)              NSGA-II search + held-out H1 comparison for one outer fold
  random_search_baseline.py       H2: same budget, uniform random genomes
tests/test_smoke.py               synthetic-data smoke test (no GPU, no medAI, no real data)
```

## Usage

```bash
pip install -r requirements.txt   # + pymoo, into the `prostNfound` conda env or the
                                   # proside-dino-pca venv -- see scripts/*.sh for which env
                                   # each step needs and why.

# 1) build the region cache (slow, GPU, run once)
sbatch scripts/cache_regions.sh                     # full ~1118 cores
sbatch --time=00:15:00 scripts/cache_regions.sh --limit 8   # smoke test first

# 2) search
sbatch scripts/run_search.sh --outer_fold 0 --pop_size 16 --n_gen 10

# 3) H2 baseline, same budget
python scripts/random_search_baseline.py --cache_dir cache/regions \
    --exp_dir experiments/random_fold0 --outer_fold 0 --pop_size 16 --n_gen 10
```

## Status

Scaffold, exercised only on synthetic data so far (`pytest tests/test_smoke.py`). Real numbers
need, in order: (1) `scripts/cache_regions.py` run for real on a GPU node -- the only slow step;
(2) `scripts/run_search.py` for at least one outer fold; (3) the H2 random-search comparison at
equal budget; (4) the H3 ablation (zero `w_negative`/`w_unlabeled` and re-run). Before committing
further to this framing: a literature search for prior work applying evolutionary loss/objective
search to MIL or PU learning in prostate ultrasound -- not yet done.
