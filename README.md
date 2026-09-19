# DB-WM — Deep Basis World Models for LST/NDVI Forecasting (JAX)

**v4 (memory-augmented, multi-horizon) is the current pipeline — see Section 0.**

> **Forecasting LST rather than NDVI? Read [`README_LST.md`](README_LST.md).**
> It is the *same* pipeline — `--modality lst`, or the `experiments/*_lst*`
> aliases — but five physical constants differ (units, display range, colour
> ramp, teacher-forcing threshold, validity gate), the LST rasters arrive on a
> geographic CRS whose pixel size is in degrees rather than metres, and the
> memory lift's requirement of `L` consecutive observed days makes the archive's
> cadence worth checking before a long run. Every one of those produced
> plausible, wrong output rather than an error.

Scalable Gaussian-process spatiotemporal forecasting of 30 m Land Surface
Temperature (LST) and NDVI over Lahore (~4,500 km², UTM Zone 43N), implemented in
**JAX / Flax / Optax**. This is the reference implementation of the framework in
[`../DB_WM_v2_framework.md`](../DB_WM_v2_framework.md) (reconciled with the
input-affine corrections of [`../DB_WM_v3_framework.md`](../DB_WM_v3_framework.md)),
specialised to the satellite-imagery application of Section 8.1.

It replaces the `O(M³)` Evolving-GP / Kernel-Observer bottleneck (the reference
`funcobspy` codebase) with the Deep Basis Kernel's low-rank structure:
`k(o,o') = ⟨φ_θ(o), φ_θ(o')⟩`, giving **`O(nr²)` training and `O(nr)` inference**
via the Woodbury identity.

---

## 0. v4: memory-augmented, multi-horizon NDVI/LST forecasting

The pipeline described from Section 1 onward is the **v2** model (`w_{t+1} = A w_t
+ B u_t`, image encoder, one-step forecast). The **v4** pipeline implements
[`../DB_WM_v3_memory_multihorizon.md`](../DB_WM_v3_memory_multihorizon.md) and is
what you should run for the Sayedanwala NDVI archive:

```bash
# Synthetic end-to-end smoke run (~40 s on CPU, no Drive needed)
python -m experiments.run_ndvi_v4 --smoke

# Real data -- only these paths need changing
python -m experiments.run_ndvi_v4 \
    --ndvi-dir    /content/drive/MyDrive/NDVI_Downscaled_v3 \
    --weather-csv /content/drive/MyDrive/Weather_v3/sayedanwala_historical_weather_2022_2026.csv \
    --r 256 --memory-order 7 --horizon 6 --epochs 50

# Then the same for LST -- the pipeline is modality-agnostic
python -m experiments.run_ndvi_v4 --modality lst --lst-dir ... --weather-csv ...
```

**That single command is the whole v4 experiment.** It runs all ten stages in
order and writes `summary.json`, `model.pkl` and `seasonal_maps.npz`. There is no
multi-step sequence to follow and no preprocessing step to run first: unlike the
v2 path, v4 builds its forcing directly from the weather CSV, so
`experiments/preprocess_forcing.py` is **not** part of it.

> **Do not run the Section 7 sequence for v4.** `train_dbwm.py`,
> `run_inference.py` and `compare_baselines.py` are the **legacy v2** scripts:
> image encoder, one-step forecast, no memory lift, no multi-horizon family, no
> weather emission, and the old 85/15 split instead of the season calendar. None of
> them import a single v4 module. They remain useful only as an explicit **baseline**
> -- "what does the one-step Markov model score on the same data?" -- which is a
> legitimate ablation to report, but it is not the thesis pipeline.

### Why the forecast RMSE was 0.10, and what fixed it

The first full run on the real record reported a pixel RMSE of **0.1026** at
`t+1`. That number is not a tuning problem. It decomposes exactly, and the
decomposition says the dynamics were never the issue:

| quantity | measured | where |
|---|---|---|
| reconstruction `R²` | 0.958 | `dbwm.gpstate` |
| ⇒ encoding error `ε_enc` | 0.184 normalised | `√(1 − R²) × spatial std` |
| `‖A_cal‖₂` | 4.13 | `dbwm.conditioning` |
| `ρ(A_cal)` | 1.0000 | `dbwm.memory` |
| **`‖A‖ × ε_enc`** | **0.1028 NDVI** | v2 Thm 4.2 at `T = 1` |
| observed `t+1` RMSE | 0.1026 NDVI | `dbwm.v4` |

Agreement to three significant figures. **The one-step error was the
representation error multiplied by the operator norm — the "dynamics error" was
essentially zero.** Two things were wrong, and both had to be fixed because
either alone leaves the product too large.

**1. The framework's own theorem was not being enforced.** v2 Theorem 4.2 opens
"let `ρ = ‖Â‖₂` be the spectral **norm**". The pipeline was constraining the
spectral **radius**. For a non-normal operator those are unrelated at finite
horizon, and here they differed by 4×: the radius sat at exactly 1.000 while the
powers peaked at 11.1. The guarantee was nominal, never in force.

The fix is not to constrain `‖A_cal‖` itself — the companion matrix carries the
shift identities, so `‖A_cal‖ ≥ 1` for any `L ≥ 2` and the constraint would be
infeasible by construction. What actually transports the error to the output is
`g_h = ‖S A_cal^h‖₂`, and `enforce_forecast_gain` bounds that. Persistence sits
at exactly `g_h = 1`, which makes it the natural reference and the natural
shrinkage anchor.

**2. A smooth coordinate MLP cannot represent an agricultural mosaic.** The AOI
is piecewise-constant plots with sharp edges, each on its own crop calendar;
Fourier features postpone spectral bias but do not remove it, and `r = 256`
plateaued at `R² = 0.958`. `dbwm/gp/empirical_basis.py` appends two data-driven
blocks — a per-pixel training climatology, and EOFs of exactly what `Ψ` leaves
behind. This is not a departure from the DBK construction: v2 Theorem 2.1 says
the ideal basis *is* the Mercer eigenfunction set, and EOFs are its empirical
estimate; the E-GP paper lists the same object among its admissible feature maps.

Three further changes follow from the same analysis:

* **increment parameterisation** (`MemoryConfig.increment`) fits `w_{t+1} − w_t`
  and adds `I` back to `A_0`. Identical model class, but the ridge now shrinks
  toward *persistence* instead of toward *zero*. On a daily NDVI record shrinking
  toward zero shrinks toward something worse than doing nothing, which is what
  forced the large blocks in the first place;
* **`calibrate_persistence_blend`** makes the operator *earn* its deviation from
  persistence on a held-out tail, with the gain cap as a hard rail. A fixed cap
  of 1.0 is too blunt — persistence already sits at 1, so it admits nothing else;
* the blend anchors on persistence rather than zero, which also removes the
  **−0.48 normalised bias** visible in the reported triptychs. Shrinking toward
  zero drags every prediction to the training mean field; that bias was the
  shrinkage, not the model.

### Three defects the first real run exposed

**1. The weather sensor was assimilated despite failing out of sample.**
`fit_emission` computed a held-out `R²` of −198 to −948, warned about it, and the
observer used the model anyway. The shared filter pass applies that update at
every one of the ~1581 steps, so it does not merely fail to help — it drives the
state each forecast starts from, producing a whole-field offset. Worse, the
per-variant `use_weather` flag could not reveal it: that toggle only affects the
forecast *branch*, while the filter pass is shared. `EmissionModel.usable` now
gates on held-out skill and the observer drops the sensor when it fails.

**2. The subspace charged its truncation error to every forecast.** The operator
is identified only inside the POD subspace, so it says nothing about the
orthogonal complement — yet `reconstruct` implicitly predicted that component to
be *zero at every horizon*. That is not neutral; it asserts the discarded
directions collapse instantly. `reconstruct_with_complement` persists them
instead, which has a sharp consequence: when the operator shrinks to persistence
inside the subspace, the forecast becomes **exactly** pixel-space persistence
rather than persistence-plus-truncation. On the real record the truncation term
was 0.021 NDVI against a persistence bar of 0.030 — two thirds of the budget
spent on a modelling artefact.

**3. `Theorem 2.8(i) FAILS` was misdiagnosing the persistence case.** When the
blend selects `s = 0`, `A_j = 0` for `j ≥ 1` *by construction*, so the
certificate reports the lifted pair as unobservable and advises reducing `L`.
That advice is wrong: there is no memory kernel to observe because the data did
not support one. `is_persistence` detects the case and says so.

### Read the error budget before chasing a target

Every run now prints this before the metrics:

```
ERROR BUDGET at h = 1 (physical units; the target must sit above this)
  representation   (y -> w -> y)          0.01450
  subspace trunc.  (w -> z -> w)          0.00633
  upstream total   (quadrature)           0.01582
  operator gain    max_h ||S A^h||        1.994x
  --> achievable floor  gain x upstream   0.03154
  persistence reference (h = 1)           0.05617
  binding term: representation.
```

The forecast **cannot** be better than `gain × upstream`, because that loss
happens before the operator is applied. If a target sits below that line, no
amount of work on the dynamics will reach it — raise `--eof-modes` (representation)
or `--subspace-energy` (truncation) first. The binding term moves once relieved,
so re-measure after each change rather than fixing all four at once.

**Persistence is the bar, and it is a high one.** On a daily, gap-filled NDVI
product the field barely moves between consecutive dates. Measure yours before
fixing an accuracy target:

```python
d = np.sqrt(np.mean((flat[1:] - flat[:-1])**2)) * ds.std[0]   # NDVI units
```

On a Sayedanwala-shaped surrogate this is ≈0.007 at `t+1` and ≈0.019 at `t+6`.
A flat "below 0.01 at every horizon" therefore asks the dynamics to roughly
*halve* the error of the best trivial forecast at `t+6` — reachable at short
lead, a genuinely strong claim at long lead. `compare_v4` always reports
persistence and climatology alongside every cell so this stays visible.

### v4 in the Section 7 shape: train / infer / ablate

`run_ndvi_v4.py` does everything in one process, which is what you want for a
single reported experiment. When you instead want to **train once and forecast many
times** — a new test window, a weather ablation, a fresh set of exported rasters —
use the split entry points. They are the v4 analogues of `train_dbwm` /
`run_inference` / `compare_baselines` and produce the same kinds of artefacts:

```bash
NDVI=/content/ndvi          # copy off Drive first -- FUSE is ~14 min for 1568 files
WX=/content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P/sayedanwala_historical_weather_2022_2026.csv

# 1. Train and checkpoint (basis, memory kernel, emission, horizon family)
python -m experiments.train_v4 --ndvi-dir $NDVI --weather-csv $WX \
    --r 256 --memory-order 7 --horizon 6 --epochs 100 --ckpt-dir ./_v4_ckpt

# 2. Forecast t+1..t+6, score, plot, and export georeferenced rasters
python -m experiments.infer_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --ndvi-dir $NDVI --weather-csv $WX \
    --export-geotiff --export-origins 20 --results-dir ./_v4_infer

# The weather-measurement ablation: B_p p_t stays, only the C-update is removed
python -m experiments.infer_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --ndvi-dir $NDVI --weather-csv $WX --no-weather

# 3. Operational forecast from ONE date (works past the archive end)
python -m experiments.forecast_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --date 2026-05-02 --ndvi-dir $NDVI --weather-csv $WX \
    --context-days 7 --export-geotiff --results-dir ./_v4_forecast

# 4. How many past days does it need? L in {2, 4, 6, 7}
python -m experiments.ablate_memory --ndvi-dir $NDVI --weather-csv $WX \
    --orders 2 4 6 7 --r 256 --horizon 6 --epochs 100

# 5. DB-WM vs E-GP / Kernel Observers, one matched-protocol run
python -m experiments.compare_v4 --ndvi-dir $NDVI --weather-csv $WX \
    --r 256 --memory-order 7 --horizon 6 --epochs 300 --eof-modes 256 \
    --egp-centers 300 --egp-meas rational
```

**`infer_v4` needs the data paths too.** It rebuilds the calendar to encode the
test frames; only the *model* comes from the checkpoint. The archive window and
split date come from the checkpoint unless you override them, and an override is
announced in the log because it can silently invalidate the train/test split.

**Every entry point shares one group of error-budget flags** (`--eof-modes`,
`--eof-energy`, `--no-climatology`, `--gain-max`, `--absolute-operator`,
`--no-teacher-forcing`), registered from `_v4_common.add_model_args` so the same
flag cannot mean different things in different scripts. Run once, read the
`ERROR BUDGET` block, then raise whichever term it names as binding.

Three things about this split are deliberate:

**The checkpoint refuses to be confused with a v2 one.** A v2 checkpoint stores a
single `A`; v4 stores `L` blocks. Loading the former without complaint would
forecast a memoryless model while every log line still claimed `L = 7` — an error
that survives all the way to a thesis table. `load_checkpoint` raises instead.

**The calendar comes from the checkpoint, not from argparse defaults.** Passing
`--split-date` to `infer_v4` overrides it, and says so in a warning, because
scoring on a window that overlaps training is a legitimate thing to want and an
illegitimate thing to do by accident.

**`--no-weather` ablates one role, not both.** Precipitation still drives the state
through `B_p p_t`; only the `y_t^w = C w_t + noise` correction is removed. An
ablation that stripped weather from the filter *and* the forcing would confound the
two and attribute the whole difference to the measurement update.

### Metrics: ubRMSE and MAE, not RMSE alone

`RMSE² = bias² + ubRMSE²` splits the error into a whole-field **offset** and
genuine **structural** disagreement. They have different causes and different
cures: a bias usually points at something specific and correctable (a
mis-specified sensor, a climatology from the wrong period, a shrinkage anchor
pulling the wrong way), while ubRMSE is what a better model actually has to
reduce. On the first real run a bias of +0.064 against an RMSE of 0.104 meant
**38% of the squared error was a constant offset** — invisible in RMSE, and
untouched by any amount of work on the dynamics.

Every reporting surface now leads with ubRMSE and MAE and keeps RMSE/bias as
context: the Stage 10 log, `summary.json`, the triptych titles, and `DBWM_*` tags
on every exported raster. Stage 10 also warns when the bias exceeds half the
ubRMSE, since that is the signature of a systematic fault rather than a modelling
limit.

### Two bugs that made rho and the predictive interval report fiction

Both were found by working backwards from a real forecast figure, and neither
raised anything — they just made reported numbers describe a different object
from the one being used.

**`rho` was the spectrum of an operator that never ran.** `lifted_spectrum` takes
a fast path from the per-mode `coefficients` whenever they exist (Prop. 2.11) and
falls back to the dense companion otherwise. Two transformations edited `blocks`
without editing `coefficients`, so the two representations drifted apart:

* `identify_memory(increment=True)` — the **default** — adds `I` to `A_0` but left
  the coefficients describing the increment `G`, so every spectral quantity was
  reported for `G` rather than for `A_0 = I + G`;
* `_blend_to_persistence` blended the blocks and carried the coefficients through
  untouched, so the `rho`-bisections could never move their own objective.

Everything downstream inherited it: the reported `rho`, the plotted Koopman
spectrum, and the Thm 2.8(ii) null directions (whose fast path also reads
`coefficients`). On a fixture that reported **`rho = 1.2769`** the operator
actually in use sat at **0.9836** — stable, inside the unit disc. Adding `I` in
modal coordinates is exactly `alpha_0 += 1`, and the persistence anchor in
coefficient space is `alpha_0 = 1, alpha_j = 0`, so both fixes are exact rather
than approximate. `test_memory.py` now pins fast-vs-dense agreement at every
blend weight.

**`rho_max` was declared and never applied.** `MemoryConfig.stability_metric`
defaults to `"forecast_gain"`, which put `enforce_stability` in an `else` branch,
so `DynamicsConfig.rho_max = 1.0` bound nothing. The gain cap bounds
`max_h ||S A^h||` over `h = 1..H` only, and a non-normal operator can satisfy it
while `rho > 1` — stable over the horizon it was scored on, divergent just past
it. `enforce_spectral_radius` now runs as *well as* the gain cap, blending toward
persistence (which sits at `rho = 1` exactly, so the bisection is always
feasible) rather than scaling toward zero.

**The predictive interval omitted its dominant term.** A pixel's forecast
variance is

```
var(s) = Psi(s)^T Sigma_w Psi(s)  +  sigma_eps^2  +  var_repr(s)
         propagated latent           observation     REPRESENTATION  <- missing
```

Only the first two were summed. The third is what the error budget already calls
the representation floor, and it is the largest of the three here. The
consequence was arithmetic, not subtle: at `t+1` the reported half-width implied
`sigma = 0.0326` against a realised ubRMSE of 0.0653 — **exactly half** — and
empirical coverage came out at 58.4% against a nominal 95%, which is precisely
what a `N(bias, ubRMSE^2)` error gives against that half-width. `uncertainty_band`
now takes `repr_var` from `representation_variance` (fitted on training dates,
carried in the checkpoint), and because that residual is spatially structured —
largest at the plot boundaries the smooth basis cannot resolve — the pixels the
model gets worst are now the pixels it says it is least sure about.

### Attributing the error instead of just measuring it

Two reports answer "where is the model still losing?", both in `error_budget.py`:

**`bias_attribution`** splits the offset by the stage that introduces it —
`encode_bias` (the basis round trip alone), `dynamics_bias` (what the filter and
operator add on top), and `drift_per_step` (slope in `h`). `RMSE^2 = bias^2 +
ubRMSE^2` says how big the offset is and never where it comes from, and the two
candidates need opposite fixes.

**`worst_pixel_attribution`** decides whether the worst pixels are *fixable*. An
error that is flat across leads on ground that barely moved is a **representation**
failure — raise `--eof-modes` or `r`. An error matched by a large *observed*
change at that pixel is an **unpredicted event**: no autonomous linear operator
driven by NDVI history and weather can know the day a parcel is harvested, so it
is not reducible by tuning `L`, `rho` or the shrinkage, and the honest response is
the wide interval it now carries rather than a better point forecast.

### The figures: fixed 0–1 scale, RdYlGn, dated, with intervals

Field panels are **pinned** rather than auto-stretched, because auto-scaling makes
a `t+1` and a `t+6` map incomparable — each fills its own range, so a forecast
that has drifted still looks fine on its own axes. Four things changed about how
they are pinned and drawn:

**The range is `[0, 1]`, not `[-1, 1]`.** NDVI is physically bounded by
`[-1, 1]`, but this field never leaves roughly `[0.10, 0.70]`. On `[-1, 1]` about
70% of the colour ramp is spent on values the scene never takes: every panel came
out as one flat wash and the plot boundaries were invisible. `[0, 1]` keeps the
scale fixed while handing the data most of the ramp. Water and bare soil clip to
the bottom colour, which is the right reading for a vegetation map. LST and any
other modality has no canonical range and falls back to robust percentiles
(`v4_plots.value_range`).

**The ramp is `RdYlGn`.** The conventional vegetation ramp — red for bare or
stressed, green for vigorous — and monotone in lightness, so greenness reads
straight off the map. `YlGn` (the old default) has almost no contrast in the
lower half of its range, which is exactly where this field sits.

**Every figure carries its dates.** Filenames are
`<name>_triptych_<origin date>_t+<h>.png`, and the panels name the origin and the
target date. A directory of `*_t+3.png` files is unusable: a lead time is only
meaningful relative to an origin.

**A fourth panel shows the predictive interval, with coverage.** It is drawn from
the same `uncertainty_band` that fills band 2 of the exported GeoTIFF, so the
figure and the raster cannot disagree. Beside it the title reports the empirical
coverage — the fraction of valid pixels with `|error| <= 1.96 sigma` — which is
how v3 Prop. 2.13's under-coverage claim becomes visible rather than assumed.

Resolution scales with the array (`_field_dpi`) so every raster cell gets at least
four device pixels; below about three, a single-pixel plot boundary is
indistinguishable from a rendering artefact. Masked pixels are drawn in an
explicit grey so "outside the field clip" cannot be misread as a pale value.

**Where the worst pixel is.** Each triptych marks and names the best- and
worst-predicted pixel. On a single date per-pixel RMSE and MAE are the *same*
statistic (`|e|`) and the figure says so instead of printing one number twice
under two names. Pooled over origins they separate, and
`<name>_pixel_error_t+<h>.png` shows per-pixel RMSE / MAE / bias maps for each
lead with the extremes marked — that is where "which pixel is worst" is a
well-posed question, and where one hot plot boundary is distinguishable from a
diffuse miss. The same numbers land in `summary.json` under `per_pixel_error`.

Pick which dates get triptychs with `--plot-date`:

```bash
python -m experiments.run_ndvi_v4 --ndvi-dir $NDVI --weather-csv $WX \
    --plot-date 2026-04-20 2026-04-24 --export-geotiff
```

### Forecasting from a specific date

```bash
python -m experiments.infer_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --ndvi-dir $NDVI --weather-csv $WX \
    --origin-date 2026-04-20 2026-04-24 --export-geotiff
```

Each requested date is validated before anything runs, and each failure mode is
reported specifically because the fixes differ: the date must be **on the
calendar and observed** (13 dates are missing — forecasting "from" one would
start the filter at a fabricated state); it must have **`L−1` observed days
behind it** (the memory lift conditions on `w_t … w_{t−L+1}`, so a gap would fill
the delay line with the filter's own predictions); and it must be **outside the
training window** unless `--allow-train-origin` is passed, since an in-sample
forecast must not be reported beside out-of-sample numbers.

Horizon steps running past 2026-04-30 are predicted and exported but excluded
from the metrics — they are what a forward forecast *is*, and scoring them
against nothing would be meaningless. A requested origin **past** the archive end
extends the calendar automatically rather than being refused, since an
operational origin is always past the end of a fixed archive.

### Operational forecasting: any date, seven days of context

`infer_v4` is retrospective — it rebuilds the whole archive to score a checkpoint
over the test split. `forecast_v4` answers the other question, the one an end user
actually asks:

> It is 2026-05-02. I have the last seven days of 30 m NDVI and the weather
> record. What does the field look like on the 3rd through the 8th?

```bash
python -m experiments.forecast_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --date 2026-05-02 --ndvi-dir $NDVI --weather-csv $WX \
    --context-days 7 --export-geotiff --results-dir ./_v4_forecast

# Fold in any horizon frames that have since arrived, correcting the rest
python -m experiments.forecast_v4 --ckpt ./_v4_ckpt/dbwm_v4_v4.pkl \
    --date 2026-05-02 --ndvi-dir $NDVI --weather-csv $WX --assimilate-future
```

It builds a **local** calendar `[date − (context−1), date + H]`, loads whatever
rasters exist in it, and refits nothing: the basis, the empirical block, the
memory kernel, the horizon family, the emission, the normalisation and the
weather climatology all come from the checkpoint. `--context-days` is raised
automatically to `L` if smaller, since the lift conditions on `w_t … w_{t−L+1}`.

Three things it refuses rather than guesses, each naming its own fix:

**The origin must be observed with `L` consecutive observed days behind it.**
Otherwise part of the delay line would be the filter's own predictions, which is
not what a conditioned forecast means.

**The pixel grid must match the checkpoint's.** The empirical (EOF) basis block is
*transductive* — defined per pixel, not as a function of coordinates — so a
different `--max-pixels` silently changes the basis. The checkpoint records
`n_pixels` and `grid_shape`, and a mismatch is an error, not a reshape.

**Nothing about the window may be estimated on the window.** A twelve-day window
fitting an annual harmonic climatology on itself would produce arbitrary
anomalies, and the anomaly is exactly what the Kalman update consumes. The
climatology and both scale vectors travel in the checkpoint
(`weather_for_window`).

Missing future weather is tolerated, and the two roles get **opposite** fallbacks:
a missing *measurement* (Rs/Ta/VPD) means no correction is available, so the row
stays `NaN` and the observer skips that update — substituting the climatology
would set the anomaly to exactly zero, which is not "unknown" but the confident
claim that the day is exactly average. A missing *input* (precipitation) means
assume no rain, which is logged, because a forecast made under an assumed-dry
horizon should be read as one.

**`--assimilate-future` and the leak it is built to avoid.** Two trajectories come
back and they are never mixed. `forecast` is what was predicted for each step
*before* that step's own frame was seen — the only one scoreable as forecast
skill. `corrected` is the filtered estimate after assimilating everything
available, and later steps are relaunched from it, so a frame at `t+2` genuinely
sharpens `t+3…t+6`. Reporting the second as forecast skill would be scoring the
model on data it has already been handed; `forecast.json` labels it accordingly.

### Ablation: how many past days does the forecast need?

```bash
python -m experiments.ablate_memory --ndvi-dir $NDVI --weather-csv $WX \
    --orders 2 4 6 7 --r 256 --horizon 6 --epochs 100 \
    --results-dir ./_v4_ablate_memory
```

v2 asserts a **Markov** transition (`w_{t+1} = A w_t + B p_t + η_t`) — the `L = 1`
truncation. v3 Prop. 2.5 shows via Mori–Zwanzig that the exact projected dynamics
*cannot* be Markov, so `L` is a real modelling choice and this measures what each
extra day of context buys. `L = 1` is a legitimate entry and is the v2 baseline
the whole extension has to beat.

**What is held fixed is the experiment, not an optimisation.** The basis `Ψ`, the
empirical block, the GP posterior, the POD subspace, the weather emission and the
evaluation origins are computed **once** and shared. v2 Thm 4.2 bounds the
`h`-step error by `ρ^h ε_enc + …`, so the representation error enters every
horizon multiplied by the operator gain — on the real record it was the *entire*
one-step error. Retraining `Ψ` per order would let that term move between cells,
and the table would be measuring basis-training variance under a column heading
that said "memory order". Origins are purged with `max(L)` for every order, so all
orders score identical dates; purging per order would hand `L = 2` five more
origins than `L = 7`.

**Three panels, because three different statements bound `L` from opposite
sides**, and a run reporting only forecast error would see none of them:

| Panel | Statement | What a failure means |
|---|---|---|
| (a) held-out ubRMSE by horizon | — | what the extra lags actually buy, against persistence |
| (b) `s_min(A_{L−1}) / ‖A_0‖` | v3 Thm 2.8(i) | over-lagging: the lift is **unobservable** and the deepest lags carry no recoverable state |
| (c) normalised `‖D_h‖` | v3 Def. 2.6 | the Mori–Zwanzig memory depth; its elbow in `L` is a physical measurement |

Parameter counts are printed beside the scores, because v3 §2.2.4's argument *is*
about counts: unstructured is `L r²` (4.6×10⁵ at `r = 256, L = 7`, against ~1,200
transitions) while S2 is `rL = 1,792`. Without them a structured win reads as an
order effect.

The report separates the **two** ways Thm 2.8(i) can fail, because they demand
opposite fixes: `A_{L−1}` singular with `A_0` well conditioned is genuine
over-lagging (reduce `L`), while `A_0` itself near-singular means `r` exceeds the
rank the trajectory excites (reduce `r`, or tighten `--subspace-energy`) and would
fail at `L = 1` too. It also flags orders shrunk all the way to persistence
(`s = 0`), whose rows measure the shrinkage anchor rather than the memory order,
and it says so when the spread across orders is below the run-to-run noise of the
basis fit — which is itself a finding.

### The head-to-head comparison, and why the grid was replaced

`compare_v4.py` used to be a 72-cell ablation over the DB-WM memory order,
parameterization, estimator and subspace flag. On the real record it ran for five
hours and could not answer the question it existed for:

**The grid axes were inert.** Every cell reported `Persistence blend selected:
s = 0.00` — the memory kernel never beat persistence on held-out data, so `A_j = 0`
for `j ≥ 1` and the memory order and parameterization had nothing to act on.
`L1_s1`, `L1_s2` and `L1_s3` returned byte-identical numbers. The table was 72
near-duplicates of one model plus noise from the `Θ_h` fit.

**The comparison was not like-for-like.** DB-WM was scored with a climatology
offset, 96 empirical modes, a persistence-blended operator and a POD subspace; the
E-GP got the raw field and none of it. A gap measured that way is protocol, not
method.

The script now runs **both methods once under one protocol** — same origins, same
valid pixels, same `field_error_metrics` — and reports the full `run_ndvi_v4`
metric set. The rows form a ladder:

| row | what it is |
|---|---|
| `climatology` | predict the training mean. The floor. |
| `persistence` | predict today's field. The bar that matters. |
| `EGP_forecast` | E-GP filtered on the **full frame** to the origin, then free-running. **The like-for-like comparator.** |
| `EGP_sensors` | same, but the filter sees only the `N` sensing pixels. The gap is what the paper's sensor economy costs. |
| `EGP_feedback` | keeps correcting *through* the horizon from the target frame. Filtering, not forecasting — labelled, never the headline. |
| `DBWM_recursive` | lifted memory filter, iterated forecast. |
| `DBWM_direct` | same filter, direct horizon family. |

`--memory-orders`, `--parameterizations`, `--estimators` and `--subspace` are gone;
the DB-WM side takes the same single configuration `run_ndvi_v4` does.

### Four defects in our E-GP, all found by making the comparison fair

None of these are in the paper — they were in our implementation, and each one
made the baseline look worse than the method is:

**1. The cyclic index was used as the sensor *count*.** Proposition 2 gives `ℓ` as
a lower **bound**. On the real record `ℓ = 1`, which left **290 of 300 centres
unobserved by any sensor**, `rank(O) = 52/300`, and reduced the "feedback"
observer to correcting a 300-dimensional state from a single pixel — which is why
AKO and FKO came out indistinguishable (0.564 vs 0.559). `sensor_rule="observable"`
now searches upward for full rank, as the paper's own Figure 9 does;
`"cyclic_index"` remains available as the literal-bound ablation.

**2. The state was re-initialised at every origin.** DB-WM filters all 1581 dates
before branching a forecast; the E-GP re-solved `w` from the origin frame alone,
entering each forecast having forgotten the history its competitor had. Fixed by
`filtered_rolling_forecast`, which runs one pass over the calendar.

**3. The bandwidth grid was absolute and hit its lower edge.** `σ = 0.05` was the
grid minimum against a 0.104 centre spacing — atoms that barely overlap. The grid
is now in units of centre spacing, and edge-of-grid selection is reported as the
failure signal it is.

**4. A whole frame was assimilated as if it were as noisy as one pixel.** The
full-field update used `σ²I` in weight space instead of the ridge posterior
`σ²(KᵀK + σ²I)⁻¹` — the same `σ_ε²Λ_X⁻¹` DB-WM assimilates. That overstated the
uncertainty of a full frame by orders of magnitude and made the full-field row
score *worse* than the N-sensor row, which cannot be right when it sees strictly
more of the same data.

**A genuine tension, reported rather than hidden.** The marginal likelihood and
the operator fit want opposite bandwidths: wider kernels fit each frame better but
drive `cond(KᵀK)` from ~1e1 at half a centre spacing to ~1e15 at two, after which
the per-step weights are not identifiable and the operator regressed on them is
degenerate. Selecting on likelihood alone picks a model whose *state cannot be
recovered*. `max_gram_condition` screens the grid for identifiability first, and
logs when it binds.

### The E-GP / kernel-observer baseline, as the paper specifies it

`dbwm/baselines/egp.py` is a reproduction of Kingravi, Maske & Chowdhary,
*Kernel Observers* (IEEE CSM, Feb 2021), cross-checked against the `funcobspy`
reference. The comparison is only worth publishing if the baseline is the method
as specified, so the paper's actual contributions are implemented rather than
approximated:

| paper element | where |
|---|---|
| dictionary-of-atoms map, eq. (2) | `rbf_features` |
| hyperparameters by marginal likelihood + median over steps | `_fit_hyperparameters` |
| per-step weights (`solve_tikhinov`, reg = `noise²`) | `_solve_weights` |
| matrix least squares for `Â`, eq. (17) | `fit` |
| shadedness, Definition 1 | `is_shaded` |
| **cyclic index `ℓ`, Proposition 2** | `cyclic_index` |
| observability rank of `O_Υ` | `observability_rank` |
| **k-invariant subspaces, Algorithm 3** | `k_invariant_subspaces` |
| **sampling locations, Algorithm 2** | `measurement_indices` |
| AKO / FKO observers, Algorithm 5 | `rolling_forecast` |

Two deliberate deviations from `funcobspy`, both documented in the module rather
than silent:

* **the operator regression.** The reference stacks `weights[0:T-1]` with a
  *duplicate* of `weights[T-1]` and regresses the full `weights` on it, pairing
  the last state with itself. That is a defect, not a design; the paper specifies
  least squares across successive weights, so `W₊` on `W₋` is what is fitted.
* **the `'rational'` measurement map.** `funcobspy` raises `NotImplementedError`.
  Since the cyclic-index bound is the paper's central theoretical claim, a
  comparison that only ever used random placement could not exercise it, so
  Algorithms 2 and 3 are implemented here.

**AKO and FKO are reported separately, and the distinction is not cosmetic.** The
feedback observer corrects with `N` sensor pixels *of the frame it is being scored
on*, so it is a filtering result; DB-WM's recursive forecast sees no NDVI after
the origin. On the smoke field FKO reaches 0.011 and AKO 0.062 — reporting the
first unlabelled beside a pure forecast would be a straightforward
misrepresentation. The honest headline comparison is **DB-WM recursive vs
`EGP_autonomous`**; FKO belongs in the table as what the E-GP does when it is
allowed to keep observing.

Complexity is left deliberately unoptimised at `O(M³)` — that cost *is* the
finding, and hiding it would misrepresent the comparison.

### GP posterior vs image encoder (DeiT / ResNet)

```bash
python -m experiments.compare_encoders --encoders gp deit resnet --epochs 50 \
    --ndvi-dir <dir> --weather-csv <csv>
```

v4 takes the state from the GP posterior; v2 took it from an image backbone. The
choice was argued on theoretical grounds in `dbwm/gp/state.py` — the encoder needs
a dense grid, so the ~52% of this bounding box outside the field clip must be
zero-filled *before the network sees it*, whereas the GP simply omits those rows —
but it had never been measured. This script measures it, holding the calendar,
split, augmented basis, dynamics identification and scoring function fixed so the
only thing that varies is how `w_t` is produced.

**Read reconstruction `R²` first, forecast RMSE second.** The encoding error
enters every horizon multiplied by the operator gain, so an encoder that
reconstructs worse cannot be rescued by better dynamics. A table that reported
only forecast RMSE would confuse the two effects.

`compare_v4` always reports **climatology** and **persistence** alongside the
learned cells. This is not decoration. On a slowly varying field persistence is
genuinely strong, and a learned forecast that does not beat it has not shown that
its *dynamics* contribute anything beyond its basis — with a non-normal operator
that is a real outcome, not a hypothetical, so the table has to make it visible.
Add `--with-egp` for the `O(M^3)` E-GP / kernel-observer predecessor, run on the
same origins and scored identically.

### What the exported GeoTIFFs contain

`--export-geotiff` writes one 3-band raster per (origin, horizon) into the source
CRS, named `NDVI_forecast_<origin date>_t+<h>.tif`:

| Band | Name | Contents |
|---|---|---|
| 1 | `NDVI_forecast` | decoded mean, physical units, de-normalised |
| 2 | `NDVI_std` | per-pixel predictive standard deviation |
| 3 | `NDVI_error` | forecast − truth, where a truth frame exists |

Pixels outside the field clip are written as the source no-data sentinel, so basis
extrapolation over the ~40% of the bounding box that was never observed cannot be
read as a prediction. `DBWM_*` dataset tags record the origin date, target date,
lead time, variant and units.

Band 2 deserves one caveat, which is also tagged in the file. The full `(r, r)`
forecast covariance is not retained — at `r = 256` over hundreds of origins that is
gigabytes nothing else reads — so only its trace survives to export time. The band
spreads that trace isotropically over the latent coordinates and lifts it through
the basis, `var(s) = (tr/k)·‖Ψ(s)‖² + σ_ε²`. Total power is exact and the spatial
structure is real (uncertainty is larger where the basis is thin), but the
anisotropy is approximated. For a calibrated band, run with `--keep-forecast-cov`.

### What changed and why

| Element | v2 | **v4** |
|---|---|---|
| Latent state | `w_t = phi(o_t)`, image encoder | `w_t = Lambda_X^{-1} Phi_X^T y_t`, GP posterior over **valid** pixels |
| Dynamics | `w_{t+1} = A w_t + B u_t` | `w_{t+1} = sum_{j=0}^{6} A_j w_{t-j} + B_p p_t` (memory order `L=7`) |
| Forecast | one step | `t+1 .. t+6`, direct family `Theta_h` with semigroup shrinkage |
| Sensors | NDVI frame only | NDVI frame (rank `r`, gappy) **+** Rs/Ta/VPD (rank 3, daily) |
| Time axis | observed dates | complete daily calendar, gaps marked unobserved |
| Split | 85/15 fraction | Kharif/Rabi/Zaid calendar, cut at 2025-04-15 |

### Data sources and column naming

```
NDVI rasters : /content/drive/MyDrive/NDVI_Downscaled_30m/NDVI_downscaled_30m
weather CSV  : /content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P
               /sayedanwala_historical_weather_2022_2026.csv
```

The archive covers **2022-01-01 .. 2026-05-01** (1,582 daily rows, no gaps, no
NaNs, no duplicates) -- one day beyond the raster calendar, which is harmless:
extra source rows are simply unused.

Two column schemas are in circulation and they do not agree, so
`dbwm/data/weather.py` resolves names through an **alias table** and the rest of
the pipeline only ever sees canonical names:

| Canonical | Sayedanwala archive | Open-Meteo collector |
|---|---|---|
| `precip_mm` | `Precip_mm` | `precip_mm` |
| `swrad_mj_m2` | `Sw_rad_mj_m2` | `shortwave_radiation_sum` |
| `ta_mean_c` | `Ta_C` | `temperature_2m_mean` |
| `vpd_mean_kpa` | `VPD_kpa` | `vapour_pressure_deficit_mean` |
| date | `Date` (M/D/YYYY) | `date` (ISO) |

`M/D/YYYY` versus `D/M/YYYY` is **not guessed**. Parseability decides it first --
`1/31/2022` can only be read one way -- and monotonicity breaks ties only in the
degenerate case where every day-of-month is 12 or lower. Guessing a locale here
would silently reorder a third of the record.

Note the archive supplies a *single* daily value per channel, so there are no
`ta_max` / `vpd_max` alternates to select.

### What the real data says about the design

| Channel | Seasonal R² | annual amp / resid | Consequence |
|---|---|---|---|
| `Ta_C` | **0.913** | 8.8x | raw Ta as a regressor **is** the Remark 6.1 collinearity failure |
| `Sw_rad_mj_m2` | 0.723 | 4.2x | same |
| `VPD_kpa` | 0.656 | 3.3x | same |
| `Precip_mm` | **0.108** | 0.79x | not seasonally dominated, so it correctly stays **raw** |

So the climatology-offset decision is not a precaution, it is required: Ta is 91%
explained by two annual harmonics, and NDVI's dominant autonomous Koopman mode is
also annual.

**70.2% of days are dry**, giving ~840 rain-free transitions in training -- the
quiescent set Algorithm 3 Stage I fits `A` on. Measurement anomalies are strongly
cross-correlated (Ta-VPD **0.749**, Rs-VPD 0.491), which is why `R` is fitted full
rather than diagonal.

### Execution model and GPU coverage

The v4 path is **not** pure JAX, and this is worth knowing before you provision a
GPU. Only one of the ten stages is JAX:

| Stage | Module | Library | GPU |
|---|---|---|---|
| 2 train `Psi` (dPPGP) | `training/gp_trainer.py` | JAX / Flax / Optax | **yes** |
| 3 GP posterior `w_t` | `gp/state.py` | NumPy + SciPy | no |
| 4 whiteness pretest | `dynamics/diagnostics.py` | NumPy | no |
| 5 memory kernel | `dynamics/memory.py` | NumPy | no |
| 6 emission `C`, `R` | `dynamics/emission.py` | NumPy | no |
| 7-8 horizon family, sweep | `dynamics/multihorizon.py` | NumPy | no |
| 9 lifted filter + forecast | `inference/lifted_kalman.py` | NumPy | no |
| 10 seasonal statistics | `evaluation/spatiotemporal_stats.py` | NumPy | no |

The NumPy choice is deliberate for stages 4-10: they are closed-form, run once,
need complex eigendecomposition (`jnp.linalg.eig` is CPU-only in JAX regardless)
and need data-dependent branching for missing observations, which `lax.scan` would
obscure for no gain.

**But note the consequence honestly:** stage 9 is the most expensive stage at
`r = 256`, and it is exactly the one the GPU cannot touch. `JAX_PLATFORMS=cuda`
accelerates basis training and nothing else. If wall-clock on stage 9 becomes the
bottleneck, porting `predict`/`update` in `lifted_kalman.py` to `jnp` is the single
highest-value change -- they are dense matmuls on `Lr x Lr` and would map cleanly.

### Performance: what actually costs time (measured, r=256, L=7)

Measured on CPU at the real configuration, **peak resident memory is ~0.6 GB**
for the smoke run and well under 2 GB for a full `r=256` run. Memory is not the
constraint; wall-clock is, and it was concentrated in three places:

| | before | after | how |
|---|---|---|---|
| `observability_certificate` | 88 s x8 calls | **0.14 s** | closed-form modal null directions |
| stage 9 filter passes | 4 x ~9 min | **1 x ~9 min** | one pass, variants branched inline |
| unread forecast covariances | 4.4 GB | **0** | `keep_cov=False` by default |

**The certificate.** Thm 2.8(ii) asks whether `C v != 0` at every root of the
matrix polynomial. Doing that by SVD at each of the `Lr` roots is `O(L r^4)` --
~3e10 operations at `r=256`. But under S1/S2 the polynomial factorises per mode,
so all `L` lifted roots of a Koopman mode share **one** spatial direction (that is
what Prop. 2.11's mode-splitting means geometrically). There are at most `K <= r`
directions to test, and they are the modal basis columns. `O(r^2)`, and exact
rather than a numerical null-space estimate.

**Origin stride.** `--forecast-stride N` scores every Nth test origin. Stage 9's
branch cost is linear in the origin count, so a stride of 3 cuts it threefold at
negligible cost to an RMSE-vs-horizon curve averaged over hundreds of origins.

**Data loading.** Reading 1,568 small GeoTIFFs through the Drive FUSE mount takes
~14.5 minutes (~0.55 s/file -- latency, not bandwidth). Copy to local disk first:

```bash
cp -r /content/drive/MyDrive/NDVI_Downscaled_30m/NDVI_downscaled_30m /content/ndvi
python -m experiments.run_ndvi_v4 --ndvi-dir /content/ndvi ...
```

That turns 14.5 minutes into ~20 seconds and is worth doing every session.

**Do you need a bigger machine?** No. A T4 with 51 GB system RAM is far more than
this pipeline needs -- and an A100 would not help, because only stage 2 uses the
GPU at all (see "Execution model and GPU coverage"). If a run is slow, the levers
are `--forecast-stride`, `--no-sweep`, and a smaller `r` -- the dynamics cost is
cubic in `r`.

### Troubleshooting: Colab CUDA plugin mismatch

Colab images often ship a CUDA plugin from a **different JAX release** than the
installed `jaxlib`. The symptom is a crash inside stage 2 with:

```
jax_cuda12_plugin version 0.7.2 is installed, but it is not compatible
with the installed jaxlib version 0.10.2
...
JaxRuntimeError: INVALID_ARGUMENT: Unexpected PJRT_FFI_UserData_Add_Args
size: expected 48, got 40. The plugin is likely built with a later version
than the framework.
```

This is environmental, not a codebase fault. The traceback points at
`jax.random.PRNGKey` because that is simply the first JAX op the pipeline
executes -- the plugin failed to initialise long before.

**Why an in-process fallback cannot rescue this.** By the time any Python code
of ours runs, `import jax` has already called `discover_pjrt_plugins()` and
registered the stale plugin. From that point every backend -- CPU included --
is created through the same PJRT layer whose ABI the plugin has broken, so
setting `jax_platforms` afterwards fails with the identical
`PJRT_FFI_UserData_Add_Args` error it was trying to escape.

The fix therefore acts **before JAX is imported at all**, and detects the problem
**without importing JAX**: `dbwm/platform.py::preflight()` compares the installed
`jax-cuda12-plugin` / `jax-cuda12-pjrt` versions against `jaxlib` (JAX ships them
in lockstep, so exact equality is the test) and, on a mismatch, sets
`JAX_PLATFORMS=cpu` so the broken backend is never initialised.

It runs from two places, so both entry points are covered:

* `experiments/run_ndvi_v4.py` -- above its `import jax.numpy as jnp`
* `dbwm/__init__.py` -- so `import dbwm.<anything>` in a notebook is protected too
  (opt out with `DBWM_SKIP_PREFLIGHT=1`)

A healthy environment is left completely untouched: no accelerator is given up
when nothing is wrong. To resolve it properly:

```bash
# (a) Keep the GPU -- install a matched set, then RESTART the runtime
pip install -U 'jax[cuda12]==0.10.2'     # must equal your installed jax version

# (b) Drop the GPU -- remove the stale plugin
pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt

# (c) Ignore the GPU for one run, no install
JAX_PLATFORMS=cpu python -m experiments.run_ndvi_v4 --smoke
```

Restarting the runtime after (a) is not optional: Colab keeps the broken plugin
loaded in the current process, so an in-process fallback cannot always take
effect.

Use `--require-accelerator` to make a broken GPU a hard error instead of a silent
downgrade -- worth setting for a long `r=256` run where you would rather fail
immediately than discover 90 minutes later that stage 2 ran on CPU.

### Complexity: what is still `O(nr^2)` and what is not

| Quantity | v2 | v4 | depends on `n`? |
|---|---|---|---|
| Build `Lambda_X` (once) | `O(n r^2)` | `O(n r^2)` | yes |
| GP posterior per date | `O(n r)` | `O(n r)` | yes |
| Decode a map | `O(n r)` | `O(n r)` | yes |
| Filter predict step | `O(r^3)` | **`O(L^2 r^3)`** | **no** |
| Horizon fit per `h` | -- | **`O((L r)^3)`** | **no** |
| Observability certificate | `O(r^3)` | **`O(L r^3)`-`O(L r^4)`** | **no** |

So the answer is split, and the split is the one that matters. **Everything that
touches pixels is still `O(nr^2)` / `O(nr)`** -- the DBK scaling claim that
motivated the whole framework is intact, and it is what lets `n` reach 10^5-10^6.
**Everything in the dynamics is independent of `n` but grew in `L` and `r`.**

That growth is inherent to the memory lift, not an implementation defect: v3
Sec. 2.2.6 states it directly ("cost rises from `O(r^3)` to `O(L^3 r^3)` per step
unless the companion sparsity is exploited"). The companion sparsity *is*
exploited here -- the lower block rows of `A_cal P` are a slice of `P` -- which
buys one factor of `L`, giving `O(L^2 r^3)` rather than `O(L^3 r^3)`.

Measured on this machine (CPU, `L = 7`, `T ~ 1200`, 1581 steps):

```
lifted predict     r=128  41 ms/step        r=256  ~330 ms/step  -> ~9 min per filter pass
horizon family     r= 96  1.3 s per fit     r=256  ~25 s per fit
memory sweep L=1..7                          r=256  ~3 min
forecast stage (375 origins x 6 h x 4 variants)  r=256  ~1 h
```

Both scale as ~8x per doubling of `r`, confirming the cubic terms. Budget roughly
**1.5 h of CPU** for a full `r = 256` run after basis training. If that is too slow,
reduce `r` before reducing `L` -- the cost is cubic in `r` but only quadratic in `L`.

### The two weather roles (do not mix them)

Precipitation is an **input**; Rs, Ta and VPD are **measurements**:

```
w_{t+1} = sum_j A_j w_{t-j} + B_p p_t + eta_t      p_t  -> B_p  (predict step)
y_t^w   = C w_t + d(doy_t) + nu_t                  Rs/Ta/VPD -> C (update step)
```

A channel that is both a known input and a measurement makes the innovation
correlated with the input, which costs the filter its minimum-variance property and
invalidates the calibration claims of v3 Prop. 2.13. Keeping them separate also
rescues Algorithm 3 Regime B: precipitation is zero on ~85% of days, so the
*quiescent* set that Stage I fits `A` on is non-empty. With always-on Ta/Rs/VPD in
`Upsilon` it would have been empty.

The day-of-year climatology `d(doy)` is carried as a **known offset** in the
emission, fitted on the training split only. Ta and Rs are near-perfect annual
sinusoids and so is NDVI's dominant Koopman mode, so leaving the cycle in the raw
channel is exactly the collinearity failure of v2 Remark 6.1.

### Why the weather sensor exists at all

During a forecast there is no NDVI frame by construction -- it is the held-out
truth. Without a second sensor the covariance grows monotonically (v2 Thm 4.2, and
at `rho_max = 1` it grows *linearly and without bound*). The daily rank-3 weather
sensor is what makes a genuine Bayesian correction possible at **every** step from
`t+1` to `t+6`. Two forecast modes are reported, because they are different
requests and neither dominates a priori:

* `recursive` -- roll the one-step lifted filter forward, updating at each step, so
  six corrections **accumulate**;
* `direct` -- predict from the origin with `Theta_h` and the directly estimated
  `Sigma_h`, then correct once. Better calibrated, but does not compound.

`fit_emission` reports per-channel **held-out R^2** and warns when it is near zero:
a rank-3 update on an `r`-dimensional state is only worth something if `C w_t`
genuinely explains part of the weather, and that has to be measured, not assumed.

### Three results the pipeline produces, not just numbers

1. **The whiteness pretest is a gate, not a formality** (v3 Sec. 2.2.0). It fits the
   `L=1` model, tests the innovations for autocorrelation, and reports the verdict
   *either way*. Non-rejection means the memory lift is pure variance inflation for
   this record and should be reported as such.
2. **Memory depth is measured, not assumed.** `||D_h||` versus `L` is a direct
   empirical measurement of the Mori-Zwanzig memory depth of the field. It is
   reported in the horizon-comparable normalisation
   `sqrt(tr(D_h Cov(w_bar) D_h^T) / tr(Cov(w_{t+h})))`, because the obvious
   `||D_h|| / ||S A^h||` inflates with `h` for any dissipative field.
3. **Over-lagging is actively harmful** (v3 Thm 2.8). If `L` exceeds the true order
   then `A_{L-1} = 0` and the lifted system is *unobservable*. The certificate
   reports `s_min(A_{L-1}) / ||A_0||`, which drops by orders of magnitude exactly at
   the true order -- so it doubles as a practical order-selector.

### Season split

Kharif 1 Jun-30 Sep, Rabi 1 Oct-end Feb (**next** year), Zaid 1 Mar-31 May. Over
2022-01-01 .. 2026-04-30 there are 4 Kharif, 5 Rabi (1 partial) and 5 Zaid (1
partial) instances. A single chronological cut at **2025-04-15** realises

```
season             train     test
Summer (Kharif)     3.00     1.00
Winter (Rabi)       3.39     1.00
Spring (Zaid)       3.49     1.17
```

-- the 3 / 3.5 / 3.5 target to within 0.11 of a season, with 1200 training and 381
test days and the test set strictly in the future. A season-blocked split would
balance the seasons exactly but, with the memory lift, lets training windows
straddle held-out blocks; that is leakage.

### Seasonal spatiotemporal statistics

`dbwm/evaluation/spatiotemporal_stats.py` emits, per season and per split: the
temporal mean field `mu(s)`, the variance `sigma^2(s)`, the coefficient of
variation `CV = sigma/mu`, the spatiotemporal covariance `C(h,u)`, and a formal
**separability** test of `C(h,u) = C_s(h) C_t(u)`.

Two traps it is built to avoid:

* **CV is unsafe on NDVI**, which crosses zero. Pixels with `|mu| < 0.05` are masked,
  the masked fraction is reported, and a robust `IQR/|median|` variant is emitted
  alongside. Returning the raw ratio would produce a map that looks informative and
  is not.
* **Centring changes what `C(h,u)` means.** Removing a global mean leaves the static
  field layout in, so `C(h,0)` mostly measures field boundaries and soil; removing
  `mu(s)` measures the dynamic anomaly covariance. Both are computed and the choice
  is a required argument.

The separability test uses a **moving block bootstrap** that resamples both time
blocks *and* pixel-pair groups. Resampling only time leaves spatial sampling noise
out of the variance estimate and pushes the false-positive rate to ~60%; with both,
it sits near nominal (1/10 on a known-separable field, 10/10 power on a known
non-separable advecting field). Note also that a temporally **white** field is
trivially separable, so `temporal_correlation` is reported to make that case visible
rather than mistakable for a finding.

### v4 module map

```
dbwm/
├── data/
│   ├── seasons.py           # Kharif/Rabi/Zaid calendar, split, coverage targets
│   ├── weather.py           # point weather split by role; forward-window forcing
│   └── ndvi_dataset.py      # daily calendar reindex, per-date masks, real coords
├── gp/
│   ├── state.py             # w_t = Lambda^-1 Phi^T y_t, Sigma_{w_t}, caching
│   └── empirical_basis.py   # climatology + residual EOFs: closes the R^2 floor
├── dynamics/
│   ├── diagnostics.py       # v3 step 0: the whiteness pretest
│   ├── memory.py            # real modal form, S1/S2/S3, companion, Lemma 2.6-2.8
│   ├── emission.py          # weather emission C, full R, information gain
│   ├── multihorizon.py      # Theta_h, semigroup shrinkage, ||D_h||, Prop 2.13
│   ├── conditioning.py      # excited rank, transient growth (why a forecast fails)
│   └── subspace.py          # POD lift: representation at r, dynamics at k
├── inference/lifted_kalman.py   # two-sensor lifted filter, t+1..t+6, assimilation
├── training/gp_trainer.py   # dPPGP training of Psi on the GP path
└── evaluation/
    ├── spatiotemporal_stats.py  # mu, sigma^2, CV, C(h,u), separability
    ├── geotiff_export.py        # 3-band forecast rasters in the source CRS
    ├── error_budget.py          # which stage is binding, and the persistence bar
    ├── forecast_figures.py      # one decode path shared by all three entry points
    └── v4_plots.py              # triptychs, per-pixel maps, spectrum, ablations

baselines/egp.py         # E-GP / kernel observers, per the 2021 CSM paper

experiments/
├── run_ndvi_v4.py       # the whole v4 experiment in one process (10 stages)
├── train_v4.py          # train + identify + certify -> .pkl        [~ train_dbwm]
├── infer_v4.py          # score over the test split, plot, export    [~ run_inference]
├── forecast_v4.py       # operational: one date in, t+1..t+6 out, correctable
├── ablate_memory.py     # memory order L in {2,4,6,7}: skill + Thm 2.8 + ||D_h||
├── compare_v4.py        # DB-WM vs E-GP, matched protocol
├── compare_encoders.py  # GP posterior vs DeiT vs ResNet, same everything else
└── _v4_common.py        # the shared calendar/weather/checkpoint contract
```

The three forecasting entry points share **one** decode path
(`evaluation/forecast_figures.py`): subspace lift with the unmodelled complement
carried forward, de-normalisation, mask, sigma band. They had already drifted
apart once — one passed the valid mask and one did not, one plotted z-scores
under a colourbar labelled NDVI — and the picture and the number must describe
the same forecast.

---

## 1. What is implemented

| Framework element | Where | Notes |
|---|---|---|
| Deep basis map `φ_θ` (backbone + expansion) | `dbwm/models/` | **Image-backbone / JEPA-style**: `w_t = φ_θ(o_t)` |
| Backbone ablations: **ResNet-18** & **DeiT** | `dbwm/models/backbones.py` | both native Flax (end-to-end JAX) |
| Expansion variants: **SwiGLU** (corrected) & **RBF** (+GELU) | `dbwm/models/expansion.py` | see §2 for the SwiGLU fix |
| Spatial DBK basis `Ψ:ℝ²→ℝ^r`, fixed-grid `Φ_Xᵀ Φ_X` cached once | `dbwm/models/spatial_basis.py`, `dbwm/gp/posterior.py` | decoder `f_t(x)=⟨w_t,Ψ(x)⟩` |
| Input-affine dynamics `w_{t+1}=A w_t + B_p p_t + B_u u_t` (§2.2) | `dbwm/dynamics/transition.py` | single input-**independent** `A` (Remark 2.1) |
| Closed-form least-squares identification (Algorithm 3) + Koopman modes | `dbwm/dynamics/identification.py` | two-stage (default) & joint; **not** a DMD (Prop. 4.4) |
| **Exogenous forcing** `p_t` (rasters) + `u_t` (CSV) → `Υ` | `dbwm/data/forcing.py`, `dbwm/data/raster_align.py` | `B=[B_p B_u]` is *learned*, see §6 |
| **Section 4 guarantees**: shadedness, observability, cyclic index, controllability, Thm 4.2 bound, Thm 4.3 DARE | `dbwm/dynamics/guarantees.py` | post-identification certificates (Remark 4.1) |
| dPPGP loss + trace reg + KL + spectral penalty (Def. 3.1) | `dbwm/training/losses.py` | anti-collapse calibration |
| **Teacher forcing** (scheduled sampling, τ=0.01) | `dbwm/training/losses.py` | see §3 |
| Training driver (Algorithm 1, end-to-end) | `dbwm/training/trainer.py` | + closed-form 2-phase path |
| Visual Basis Observer = Kalman filter in ℝ^r (Algorithm 2) | `dbwm/inference/kalman.py` | predict-then-correct |
| **CEM planning** over irrigation (Algorithm 2, PLAN block) | `dbwm/inference/planning.py` | rain/irrigation mutual exclusivity as a box constraint |
| Forecasting + per-pixel uncertainty maps | `dbwm/inference/observer.py` | open- & closed-loop |
| **E-GP / Kernel Observer baseline** (JAX port of `funcobspy`) | `dbwm/baselines/egp.py` | `O(M³)`, for comparison |
| Metrics (RMSE/MAE/R²/SSIM/NLL) + publication plots | `dbwm/evaluation/` | |
| Experiment scripts | `experiments/` | preprocess-forcing / train / infer / compare |
| Tests (198: 65 v2 + 133 v4) | `tests/` | all pass on CPU |

### Design choice: image encoder + spatial GP decoder
Per the requested architecture (ResNet/DeiT backbones, Section 8.1), a whole
frame `o_t` is **encoded** to the latent weight `w_t = φ_θ(o_t) ∈ ℝ^r` (the
world-model state the linear dynamics evolve). The LST/NDVI map is **decoded**
through the spatial Deep Basis Kernel `Ψ:ℝ²→ℝ^r` as `f_t(x)=⟨w_t, Ψ(x)⟩`. Because
the pixel grid is identical across all dates, `Φ_X∈ℝ^{n×r}` (rows = `Ψ(pixel)`)
is built once and `Φ_Xᵀ Φ_X ∈ ℝ^{r×r}` cached — this is what delivers the
`O(nr²)`/`O(nr)` complexity and the calibrated per-pixel variance
`‖Lᵀ Ψ(x)‖² + σ_ε²`.

---

## 2. The corrected SwiGLU formula

`DB_WM_v2_framework.md` §2.4 wrote the SwiGLU expansion in an abbreviated
single-projection form. The **correct** GLU-variant (Shazeer 2020, *GLU Variants
Improve Transformer*) uses **two** linear projections with a Swish gate:

```
SwiGLU(x) = Swish_β(W₁x + b₁)  ⊙  (W₂x + b₂),     Swish_β(z) = z · σ(βz)
φ_θ(o)    = diag(s) · SwiGLU(g_θ(o))
```

implemented in `dbwm/models/expansion.py::SwiGLUExpansion` (verified against the
formula in `tests/test_expansion.py`). The RBF variant is left mathematically
faithful to the framework (sparse-DKL inducing-point whitening), with a
numerically stable Cholesky whitening and dimension-aware bandwidth.

---

## 3. Teacher forcing (training) & predict-then-correct (inference)

**Training (your spec).** During each trajectory rollout the model predicts
`ŵ_{t+1}=A w_t (+ B u_t)` and the per-step error `‖ŵ_{t+1} − w_{t+1}‖²` is
compared to the threshold **τ = 0.01**. If it exceeds τ, the *ground-truth*
`w_{t+1}` is fed into the dynamics for the next step instead of `ŵ_{t+1}`
(scheduled sampling). This curbs error accumulation and is implemented
differentiably with `jax.lax.scan` + `jnp.where` (the branch selector is
`stop_gradient`'d). See `dbwm/training/losses.py::teacher_forced_rollout`.

**Inference (your spec).** At each test step the observer first **predicts** with
its prior (the dynamics) and then **updates** that prediction using the held-out
LST/NDVI frame as a measurement (encoded observation = identity measurement map).
See `dbwm/inference/kalman.py::filter_sequence` and
`dbwm/inference/observer.py::closed_loop_filter`.

### Teacher forcing in v4 (schedule (b), τ = 0.02 NDVI)

**Implemented and on by default.** During basis training the latent state is
rolled forward `K` steps through a trainable operator, decoded to pixels, and
scored against the true frames. Whenever that decoded RMSE exceeds `τ`, the
**ground-truth** state is fed into the next step instead of the model's own
prediction — scheduled sampling, exactly as specified. See
`dbwm/training/gp_trainer.py::teacher_forced_rollout`, controlled by
`TrainingConfig.rollout_steps`, `lambda_rollout` and `tf_threshold_ndvi`;
`--no-teacher-forcing` recovers the previous behaviour.

Two implementation points matter:

**The threshold is applied in pixel space, not latent space.** `0.02` is an NDVI
number, and `‖ŵ − w‖` is not commensurate with it: latent distance depends on the
arbitrary scaling of the basis, so the same threshold would mean different things
for different `Ψ` and would silently change meaning *as `Ψ` trains*. Decoding
first makes the criterion exactly the reported metric, so "forced whenever the
forecast is worse than 0.02 NDVI" is literally true. The config value is in
physical units and is divided by the frame normalisation internally.

**Rollout windows must be gap-free.** The calendar has 13 missing dates; rolling
across a hole and scoring the result as a one-day step would silently mis-state
the dynamics, so only runs of consecutive observed dates are used. The training
log reports how many such windows exist and what fraction of steps were forced —
if `forced` stays at 100%, the model is never running on its own output and the
term is doing nothing but reconstruction.

`v2` has its own version at `losses.py::teacher_forced_rollout` (threshold 0.01,
latent-space), used by the legacy encoder trainer.

**This is v2 Remark 6.1 schedule (b), and it carries a known risk.** v3 Open
Item 2 warns that a multi-horizon loss rewards features that are merely
*predictable* rather than informative — an anti-collapse pressure dPPGP was not
designed for. The term is therefore weighted rather than dominant, and the
weight-trajectory rank is reported every run so a collapse would be visible as a
sudden drop in effective rank.

The closed-form counterpart is still there and still the default for the
*dynamics*: `Θ_h` is fitted on `(w̄_t → w_{t+h})` pairs and shrunk toward `𝒮𝒜^h`
by `ν_h`, which is the same bias/variance trade-off made continuous — `ν_h → ∞`
is free-running, `ν_h = 0` fully forced, and v3 Thm 2.12 puts the optimum in the
interior. Both endpoints are reportable via `--estimator iterated` / `direct`.

---

## 4. Installation

```bash
cd dbwm_v2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # JAX/Flax/Optax/rasterio/...
pip install -e .                         # optional: install as a package
```

**This describes the v2 path only.** The v2 GP/dynamics core is pure JAX and
runs identically on CPU and GPU. The **v4 path is a deliberate hybrid** and the
claim does NOT carry over: only basis training is JAX/GPU-accelerated, while the
whole dynamics/inference chain is NumPy and CPU-only. See Section 0,
"Execution model and GPU coverage".

Backbones are **native Flax** (no PyTorch in the training loop), which avoids the
JAX↔PyTorch device/jit conflicts that arise when running an eager PyTorch forward
inside a jit'd step. PyTorch/HF `transformers` are therefore **optional** and
only needed if you want to port pretrained DeiT weights (see §8).

---

## 5. Data layout (Google Drive)

Default paths assume a Colab-mounted Drive (`drive.mount('/content/drive')`):

```
/content/drive/MyDrive/
├── LST_Downscaled_v3/        # single-band LST GeoTIFFs   (30 m, UTM 43N)
├── NDVI_Downscaled_v3/       # single-band NDVI GeoTIFFs  (30 m, UTM 43N)
├── Precipitation_v3/         # daily precipitation rasters (CHIRPS / IMERG / ERA5)
└── Irrigation_v3/
    └── irrigation_schedule.csv   # date,irrigation_mm
```

Filenames must contain the acquisition date (`YYYY-MM-DD`, `YYYYMMDD`, or
`YYYY_DOY`) so frames are ordered chronologically. The loader resamples each tile
to `image_height × image_width` (default 256², configurable), masks no-data,
normalises with **training-split statistics only**, and splits **85 % / 15 %
chronologically** (test set is strictly in the future). Override the folders with
`--lst-dir` / `--ndvi-dir`, or run on the built-in **synthetic generator** with
`--synthetic` (or `--smoke`) — no Drive required.

**LST and NDVI are separate models** (separate runs, separate checkpoints), each
built against its own acquisition calendar.

---

## 6. Exogenous forcing: `B = [B_p  B_u]`

`B` is never supplied — only the regressor `Υ` is. Algorithm 3 Stage II *learns*
the forcing directions:

```
[B_p  B_u] = ΔW · Υᵀ (Υ Υᵀ + μ I_ℓ)⁻¹      ΔW_t = w_{t+1} − A w_t
```

| Component | Role | Source → representation |
|---|---|---|
| `p_t` precipitation | **uncontrollable disturbance** | GeoTIFF rasters → warped onto the LST/NDVI grid, accumulated (mm), reduced to `J_p` AOI/zonal scalars, + lag channels |
| `u_t` irrigation | **controllable actuator** | CSV time series → aggregated onto the same acquisition windows, `J_u` scalars |
| `B_p`, `B_u` | *learned*, `r`-vectors | Algorithm 3 Stage II (above) |

**Precipitation alignment.** CHIRPS (0.05°) / IMERG (0.1°) / ERA5 (0.25°) live on a
different CRS, resolution and extent from the 30 m UTM-43N imagery, so each raster
is fully **warped** (reproject + resample) onto the reference grid read from the
imagery itself — pixel-by-pixel coincident, not array-resized (which would
mis-register by kilometres). It is then reduced to scalars, so this is
*registration*, not rainfall downscaling.

**⚠ Temporal index convention (load-bearing).** Since `w_{t+1} = A w_t + B_p p_t +
B_u u_t`, row `t` of the forcing is the water arriving in the **forward** window
`(d_t, d_{t+1}]` — the interval it *drives*. The framework fixes this three times
over: Algorithm 3 pairs `Υ`'s row `t` with the residual `w_{t+1} − A w_t`;
Algorithm 2 predicts with `u_{t-1}`; and §8.1 calls `p_t` the "known precipitation
**forecast**" for `f_{t+1}`. Accumulating *backwards* into `(d_{t-1}, d_t]` — rain
already baked into `w_t` — shifts every channel one step late and corrupts `B`.
Antecedent rain is not lost: that is exactly what the **lag channels** carry
(lag-1 at step `t` = the wet soil already present when `w_t` was observed).

Because the raster warp is the expensive step and depends on no model parameter,
run it **once per modality** and cache:

```bash
python -m experiments.preprocess_forcing --modality lst   # -> forcing_lst.npz
python -m experiments.preprocess_forcing --modality ndvi  # -> forcing_ndvi.npz
```

It also reports the **identifiability** diagnostics of Remark 6.1 — the quiescent
fraction (Stage I fits `A` on rain-free, irrigation-free transitions) and
channel collinearity (monsoon rain aligned with the seasonal swing is the failure
mode that makes `A` and `B_p` inseparable).

---

## 7. Running

All commands are run from `dbwm_v2/`.

```bash
# Quick CPU smoke run (synthetic data + synthetic sparse forcing, ~1 min)
python -m experiments.train_dbwm    --smoke
python -m experiments.run_inference --smoke
python -m experiments.compare_baselines --smoke

# 1. Cache the forcing table for this modality (one-off raster warp)
python -m experiments.preprocess_forcing --modality lst

# 2. Train (LST, ResNet backbone, SwiGLU expansion)
python -m experiments.train_dbwm \
    --modality lst --backbone resnet --expansion swiglu \
    --r 512 --epochs 50 \
    --forcing-path forcing_lst.npz \
    --lst-dir /content/drive/MyDrive/LST_Downscaled_v3

# Pure-temporal ablation (B_p = B_u = 0, the §8.1 special case)
python -m experiments.train_dbwm --modality lst --no-forcing

# 3. Inference / forecasting (triptych, RMSE curve, Koopman spectrum)
python -m experiments.run_inference --ckpt <ckpt>.pkl

# 4. Ablation grid + E-GP baseline
python -m experiments.compare_baselines \
    --modality lst --backbones resnet deit --expansions swiglu rbf --epochs 50
```

Repeat with `--modality ndvi` for the NDVI dataset.

> **These four commands are the legacy v2 sequence.** For the v4 model the
> equivalents are `train_v4.py`, `infer_v4.py` and `compare_v4.py` (§0), which take
> the same shape but carry the memory lift, the horizon family and the season
> calendar. Step 1 has no v4 counterpart: v4 builds its forcing straight from the
> weather CSV, so there is no raster warp to cache.

Training prints the **Section 4 certificates** after identification — observability
rank, cyclic index, spectral radius *and norm*, the Theorem 4.2 open-loop bound,
the Theorem 4.3 steady-state `tr(P_∞)`, and the controllability rank of the
irrigation channel alone.

---

## 8. Hardware & performance

| Stage | Recommended HW | Notes |
|---|---|---|
| Training (`r`=512, 256² imgs, ResNet/DeiT) | **1× GPU** (T4/V100/A100, ≥12 GB) | the backbone forward/backward dominates; GP ops are `O(br²)` per step |
| Closed-form DMDc identification | CPU or GPU | seconds; `O(T(r+ℓ)² + (r+ℓ)³)` |
| Inference / Kalman / forecasting | CPU is fine | all ops `O(r²)–O(r³)`, independent of pixel count |
| Decoding full maps (`n`≈5.85 M px) | GPU helps | `O(nr)` per map, streamed in pixel batches |

Everything runs on **CPU** too (all tests + smoke runs here were CPU-only); GPU
is recommended only to make the image-backbone training fast. Select the device
with `JAX_PLATFORMS=cpu` / `JAX_PLATFORMS=cuda`. Memory stays `O(nr)` because the
full `n×r` feature matrix is never materialised (Gram accumulated in pixel
batches; maps decoded in coordinate chunks).

---

## 9. Optional: pretrained DeiT weights

The default DeiT backbone trains from scratch in Flax (architecturally a ViT with
a distillation token; `dbwm/models/backbones.py::DeiTBackbone`). To initialise
from HuggingFace `facebook/deit-tiny-distilled-patch16-224`, load the PyTorch
weights **offline** (one-time, outside the training loop) and copy them into the
matching Flax parameter tree — the patch-embed conv, positional embedding,
CLS/DIST tokens, and the per-block attention/MLP/LayerNorm names line up
one-to-one. This keeps training pure-JAX while reusing ImageNet features.

---

## 10. Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -q     # 358 tests, ~3 min on CPU
```

Coverage: SwiGLU formula correctness & shadedness; Woodbury posterior vs the naive
`O(n³)` GP; least-squares operator recovery (joint & two-stage) and eigenvalue
clipping; Kalman predict/update/filter behaviour; data split = 85/15 chronological
+ normalisation; empirical `O(n)`-in-`n` / `~r²`-in-`r` scaling of GP inference;
and the full train→identify→infer pipeline for both backbones, both expansions,
and the E-GP baseline.

Plus, specific to the forcing and control machinery:

* **`test_forcing_alignment.py`** — pins the forward-window convention
  `(d_t, d_{t+1}]`, and asserts that a deliberately one-step-shifted forcing
  *destroys* the recovered `B`. This is the regression test for the off-by-one
  described in §6: it is silent (shapes match, `B` is still non-zero) and would
  otherwise only show up as an inexplicably weak precipitation response.
* **`test_guarantees.py`** — Def. 4.1 shadedness, Thm 4.1 observability rank,
  Cor. 4.1 cyclic index, Prop. 4.3 controllability (incl. the honest *negative*
  result that one irrigation scalar cannot control an `r`-dimensional state),
  Thm 4.2 in both the `ρ=1` (linear) and `ρ≠1` (geometric) regimes, and that the
  Thm 4.3 `P_∞` really is a Riccati fixed point.
* **`test_planning.py`** — CEM never irrigates into forecast rain (enforced by
  construction, not by penalty); lagged rain does *not* veto irrigation; and the
  covariance is provably unchanged by the known forcing (Thm 4.3).

And specific to the v4 output layer:

* **`test_geotiff_export.py`** — georeferencing survives the round trip; invalid
  pixels stay no-data so basis extrapolation is never readable as a prediction; and
  the uncertainty band conserves total power. That last one is the substantive
  check: broadcasting `sqrt(trace)` flat across the raster, the obvious
  implementation, overstates the per-pixel deviation by `sqrt(k)` and erases the
  spatial structure the basis carries.
* **`test_v4_checkpoint.py`** — a v2 checkpoint is *refused*, not silently loaded as
  `L = 1`; and every v4 sub-config (`weather`, `seasons`, `memory`, `horizons`)
  survives the round trip, since dropping one reverts it to defaults and mis-shapes
  `B_p` against `p_t`.
* **`test_v4_plots.py`** — every figure entry point renders a real file, including
  the degenerate cases that occur in practice: a horizon with no scorable origin
  (NaN), and an ablation grid where every cell failed. Plus the choices that
  carry meaning: field panels are pinned to `[0, 1]` and the error panel stays
  symmetric about zero; the dates and the extreme pixels reach the title; a
  forecast with no truth drops the observed and error panels rather than
  inventing a frame; and per-pixel RMSE and MAE, pooled over origins, really do
  rank pixels differently — a pixel with one huge miss beats a steady one on RMSE
  and loses on MAE, which is the whole reason to aggregate.
* **`test_forecast_v4.py`** — the operational path. The assimilating loop must
  reproduce the plain forecast exactly when nothing arrives, must leave the
  forecast at and before an assimilated step bit-identical (this is where a leak
  would hide), and must improve the steps that follow it — averaged over many
  origins, not judged on one draw of the process noise. Operational weather must
  use the supplied climatology rather than refitting on twelve days, and must
  leave uncovered dates `NaN` in the measurement while forcing them to zero
  precipitation. And `_blocking` must return tile sizes GDAL accepts: the
  Sayedanwala grid is 135×125 and 125 is not a multiple of 16, so passing the
  raster width through — as the writer used to — raised `RasterBlockError` before
  a single file was written.
* **`test_ablate_memory.py`** — every order is scored on identical origins
  (purging per order silently invalidates the comparison), and the parameter
  counts follow the structures v3 §2.2.4 quotes: `rL` for S2 against `L r²`
  unstructured.
* **`test_empirical_basis.py`** — the augmentation raises reconstruction `R²`; the
  modes are fitted on training dates only (perturbing the test period must not
  move the basis — otherwise every forecast metric improves invisibly for the
  wrong reason); the empirical block is orthogonal to the learned one.
* **`test_forecast_gain.py`** — persistence has gain exactly 1 at every horizon;
  a non-normal operator with `ρ ≤ 1` is *caught* by the gain and missed by the
  radius (the real failure, in miniature); blending at `s = 0` yields persistence
  and not zero; and the increment and absolute parameterisations coincide when
  unregularised, so the reparameterisation moves the prior, not the model class.
* **`test_teacher_forcing.py`** — a perfect operator is never forced and a
  hopeless one always is; the threshold is applied *after* adding the climatology
  back, or every step would force forever; and gradients survive the scan.
* **`test_end_to_end.py::test_egp_*`** — the feedback observer must beat the
  autonomous one (the paper's own headline result; if that ordering ever inverts
  the measurement update is wired wrong and the comparison silently flatters
  DB-WM), and the default sensor count is the cyclic index.

---

## 11. Notes on the comparison numbers

In a 2-epoch **synthetic smoke run** the E-GP baseline can look stronger than
DB-WM — this is expected and **not** a real result: E-GP solves an *exact*
per-frame ridge regression with direct access to sensor pixels and needs no
training, whereas DB-WM's encoder/decoder are barely trained after two epochs on
32² images. For a meaningful comparison train DB-WM for the full `--epochs`
schedule on the real GeoTIFFs; DB-WM's advantages — `O(nr²)` scaling to millions
of pixels, learned nonstationary kernel, calibrated uncertainty, and a
Koopman-interpretable operator — appear at the real data scale where the E-GP's
`O(M³)` cost and stationary kernel become limiting.

---

## 12. Repository map

```
dbwm_v2/
├── dbwm/
│   ├── config.py              # all dataclass configs (+ smoke_config)
│   ├── data/
│   │   ├── geotiff_dataset.py # GeoTIFF loading, 85/15 chronological split, pixel grid
│   │   ├── raster_align.py    # warp precipitation onto the LST/NDVI reference grid
│   │   └── forcing.py         # p_t + u_t -> Upsilon  (forward-window convention!)
│   ├── models/                # backbones, expansions, spatial basis, variational, DBWM
│   ├── gp/                    # Woodbury low-rank posterior
│   ├── dynamics/
│   │   ├── transition.py      # A, B; spectral radius/norm, eigenvalue clipping
│   │   ├── identification.py  # Algorithm 3: least-squares [A B], Q, Koopman modes
│   │   ├── guarantees.py      # Section 4 certificates (observability, DARE, bounds)
│   │   └── dmdc.py            # deprecated shim (the fit is NOT a DMD; Prop. 4.4)
│   ├── training/              # losses (+teacher forcing), trainer
│   ├── inference/
│   │   ├── kalman.py          # Algorithm 2: weight-space Kalman observer
│   │   ├── observer.py        # map forecasting, open- & closed-loop
│   │   └── planning.py        # Algorithm 2 PLAN block: CEM over irrigation
│   ├── baselines/             # E-GP / Kernel Observer (JAX)
│   └── evaluation/            # metrics, plots
├── experiments/               # preprocess_forcing, train_dbwm, run_inference, compare_baselines
├── tests/                     # pytest suite (65)
├── requirements.txt
└── setup.py
```
