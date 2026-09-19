# DB-WM v4 for LST — multi-step-ahead Land Surface Temperature forecasting

Companion to [`README.md`](README.md), which documents the NDVI pipeline. **This
is the same pipeline.** Read that file for the model, the theory, the error
budget and the diagnostics; read this one for what is different about LST, what
commands to run, and the decisions that were taken on your behalf.

---

## 0. The one-line summary

There is no second codebase. `run_lst_v4`, `train_lst_v4`, `infer_lst_v4`,
`forecast_lst_v4`, `ablate_memory_lst`, `compare_lst_v4` and
`compare_encoders_lst` are thin aliases that force `--modality lst` over the
identical implementation; what genuinely differs between a vegetation index and a
surface temperature is **data**, in `dbwm/data/modality.py`, not code.

That was a deliberate choice over duplicating the 1,100-line pipeline. Two copies
would start identical and end different, silently — both would still run — and
the whole reason for running both modalities is that their numbers are
comparable. `tests/test_lst_modality.py` pins the equivalence:
`run_lst_v4 <args>` ≡ `run_ndvi_v4 --modality lst <args>`.

---

## 1. Commands

Every NDVI command has an LST counterpart, argument for argument.

```bash
# Synthetic end-to-end smoke run (~40 s on CPU, no Drive needed)
python -m experiments.run_lst_v4 --smoke

# Copy off Drive first -- FUSE is ~14 min for ~1,570 small files, ~20 s locally
cp -r /content/drive/MyDrive/LST_Downscaled_30m /content/lst
LST=/content/lst
WX=/content/drive/MyDrive/Historical_Dataset_SWR_VPD_Ta_P/sayedanwala_historical_weather_2022_2026.csv

# 1. The whole experiment in one process (ten stages, summary.json + figures)
python -m experiments.run_lst_v4 --lst-dir $LST --weather-csv $WX \
    --r 256 --eof-modes 512 --memory-order 7 --horizon 6 --epochs 1000 \
    --subspace-energy 0.99

# 2. Train once, checkpoint
python -m experiments.train_lst_v4 --lst-dir $LST --weather-csv $WX \
    --r 256 --memory-order 7 --horizon 6 --epochs 1000 \
    --ckpt-dir /content/drive/MyDrive/dbwm_v2/_v4_ckpt_lst

# 3. Score the test split, plot, export georeferenced rasters
python -m experiments.infer_lst_v4 \
    --ckpt /content/drive/MyDrive/dbwm_v2/_v4_ckpt_lst/dbwm_lst_gp_swiglu_v4.pkl \
    --lst-dir $LST --weather-csv $WX \
    --export-geotiff --export-origins 20 \
    --results-dir /content/drive/MyDrive/dbwm_v2/_v4_infer_lst

# 4. Operational forecast from ONE date (works past the archive end)
python -m experiments.forecast_lst_v4 \
    --ckpt /content/drive/MyDrive/dbwm_v2/_v4_ckpt_lst/dbwm_lst_gp_swiglu_v4.pkl \
    --date 2026-05-02 --lst-dir $LST --weather-csv $WX \
    --context-days 7 --export-geotiff

# 5. DB-WM vs E-GP / kernel observers, matched protocol
python -m experiments.compare_lst_v4 --lst-dir $LST --weather-csv $WX \
    --r 256 --memory-order 7 --horizon 6 --epochs 600 --eof-modes 256 \
    --egp-centers 300 --egp-meas rational

# 6. GP posterior vs DeiT vs ResNet for the latent state
python -m experiments.compare_encoders_lst --lst-dir $LST --weather-csv $WX \
    --encoders gp deit resnet --r 256 --epochs 1000

# 7. Memory-order ablation: how many past days does the forecast need?
python -m experiments.ablate_memory_lst --orders 2 4 6 7 \
    --lst-dir $LST --weather-csv $WX

# 8. The "before" ablation -- no empirical basis, absolute operator, no forcing
python -m experiments.run_lst_v4 --lst-dir $LST --weather-csv $WX \
    --eof-modes 0 --absolute-operator --no-teacher-forcing \
    --results-dir /content/drive/MyDrive/dbwm_v2/_before_lst
```

`--modality lst` on the plain entry points is exactly equivalent and remains
supported. The aliases exist so a command reads as what it does — and
`infer_lst_v4` / `forecast_lst_v4` additionally **refuse an NDVI checkpoint**
before doing any work, rather than producing NDVI results under an LST command.

---

## 2. What is identical, and why that is the point

Nothing in the mathematics changes. The GP posterior
`w_t = Λ_X⁻¹ Φ_Xᵀ y_t`, the memory lift `w_{t+1} = Σ_j A_j w_{t−j} + B_p p_t`,
the semigroup-shrunk horizon family `Θ_h`, the lifted two-sensor Kalman filter,
the season calendar and split at 2025-04-15, the observability certificates, the
whiteness pretest, the error budget — all of it operates on `w_t ∈ ℝ^r` and never
looks at what a pixel means. `L = 7`, `H = 6`, `t+1 … t+6`: unchanged, as
specified.

The weather record is the same file, with the same two roles:

```
w_{t+1} = Σ_j A_j w_{t−j} + B_p p_t + η_t     precipitation  -> B_p (predict step)
y_t^w   = C w_t + d(doy_t) + ν_t              Rs / Ta / VPD  -> C   (update step)
```

**This split is more important for LST than it was for NDVI, not less.** Air
temperature, shortwave radiation and VPD are the forcing terms of the surface
energy balance that *sets* LST, so `C` should explain a far larger share of the
state than it did for a vegetation index, and the held-out `R²` that
`fit_emission` reports is the number to read first. Precipitation stays an input
rather than a measurement for the same structural reason as before — a channel
that is both makes the innovation correlated with the input and costs the filter
its minimum-variance property (v3 Prop. 2.13) — and physically it still belongs
there: rain wets the surface and the latent-heat flux drops LST by several kelvin
within a day.

The day-of-year climatology offset `d(doy)` is likewise *more* necessary here.
The README's table shows `Ta_C` is 91% explained by two annual harmonics; LST is
very nearly `Ta` plus a surface term, so its dominant autonomous Koopman mode is
annual too. Regressing raw `Ta` on the raw state is exactly the v2 Remark 6.1
collinearity failure, and it would be *worse* for LST than for NDVI.

---

## 3. What differs, and what each difference costs if ignored

Five constants differ. All five were NDVI literals before this work, every one of
them produced output that **looked fine**, and none of them raised.

| | NDVI | LST | What ignoring it does |
|---|---|---|---|
| units | index | **kelvin** | every metric read on the wrong scale |
| display range | `[0, 1]` | **`[283.15, 320.15] K`** | a 310 K field on `[0,1]` saturates: every panel one flat wash |
| field ramp | `RdYlGn` | **`inferno`** | green reads as "healthy" where it means "cool" |
| teacher-forcing τ | `0.02` | **`0.5 K`** | 0.02 K is 1/85 of the field's σ: every step forced forever, log still says "on" |
| validity gate | none | **`[283.15, 320.15] K`** | one extrapolated pixel moves the training mean/std every frame is standardised by |

Plus two geometry/statistics items, covered in §4 and §5.

### Metrics: ubRMSE, MAE, RMSE, bias — all in K

`RMSE² = bias² + ubRMSE²` and the reasoning behind leading with ubRMSE are
unchanged; only the unit is. Every reporting surface now carries it: the Stage 10
log, the `ERROR BUDGET` header, the triptych titles, the colourbars, the
`DBWM_*` raster tags, and a top-level `"units": "K"` in `summary.json` so a
results file can never be read on the wrong scale. Precision drops from four
decimals to three — four decimals on an error of ~1 K prints noise.

`R²` is dimensionless and unchanged, but note it is **flattered** on LST relative
to NDVI: the denominator is the field's total variance, and an LST field with a
large annual swing gives a large denominator. Compare ubRMSE against the
persistence bar, not `R²` against the NDVI run.

### LST is kept in kelvin, not converted to °C

The conversion is a constant offset, so RMSE, ubRMSE, MAE and bias are *identical*
in K and °C — there is no accuracy argument either way. What decides it is that a
forecast raster silently in different units from the input raster beside it in
QGIS is a real hazard, and the archive is in kelvin.

---

## 4. The LST rasters are on a different CRS — and that is a correctness issue

From the reference tile report:

| | NDVI archive | LST archive |
|---|---|---|
| CRS | EPSG:32643 (UTM 43N, **projected**) | EPSG:4326 (**geographic**) |
| pixel size | 30.0 **m** | 0.00026949 **degrees** |
| grid | 135 × 125 | 136 × 146 |
| valid fraction | 48.4% | 48.1% (same field clip) |
| no-data | −9999.0 | −9999.0 |

The pixel size was being read as `abs(transform.a)` and used as a **metre** count.
On a projected archive that is correct and the distinction never arose; on the LST
archive it understates every ground distance by ~10⁵. That number is the bin width
of the spatiotemporal covariogram `C(h, u)`, so the separability test would have
been computed against a nonsense distance axis.

`ground_sample_distance()` now converts using the local ellipsoid scale at the
tile's latitude. At 31.08° N a square 0.000269° pixel is **~25.7 m E-W and
~30.0 m N-S** — exactly the anisotropy the raster report quotes — and the
equal-area geometric mean (~27.8 m) is what the isotropic distance bins get. The
conversion is logged, with the anisotropy stated, so it is visible rather than
assumed.

Two consequences worth knowing:

* **The basis is unaffected.** `make_coordinate_grid` normalises each axis to
  `[−1, 1]` independently, so `Ψ` sees the tile as a unit square either way.
* **If your LST archive is actually EPSG:32643** (i.e. you reprojected it to match
  the NDVI grid), nothing here fires and everything behaves as for NDVI. The code
  detects the CRS rather than assuming one.

---

## 5. Coefficient of variation is not meaningful for LST

`CV = σ/µ` is reported per season for both modalities so the table keeps one
layout, but on an **absolute** temperature scale it is not a normalisation: µ is
~300 K by construction, so every pixel returns ~0.005 and the map measures the
kelvin offset rather than the field's variability. A reader comparing that against
a 30% NDVI CV would conclude LST is a hundred times steadier, which is an artefact
of the zero point.

The run logs the caveat explicitly and the guidance is: **read the variance (or σ)
panel**. The NDVI zero-crossing guard (`|µ| < 0.05`) is inapplicable and never
fires on kelvin — if it ever does, the frames are not in the units the run thinks
they are, and the warning now says so.

---

## 6. The one thing you must check before a real run: cadence

This is the single largest risk in porting the pipeline, and it is a property of
**your archive**, not of the code.

The memory lift conditions on `w_t … w_{t−L+1}` being **L consecutive days**
(v3 Def. 2.3), and a forecast origin additionally needs `L−1` observed days behind
it plus a full horizon ahead. The NDVI archive is daily gap-filled — 1,568 frames
over 1,581 days — so at `L = 7` this is nearly free.

If the LST archive is at native satellite revisit rather than daily gap-filled,
almost no date qualifies and Stage 9 would score **zero origins** after an hour of
basis training. The loader now measures this up front and says so:

```
WARNING  LST coverage is only 6.2% of the daily calendar (98 of 1581 days,
         longest observed run 1 days). The memory lift needs L consecutive
         observed days per origin, so at this density most origins will be
         rejected. ...
```

The reference tile is named `LST_downscaled_30m_2022-04-09_sayedanwala.tif` and is
a MoCoLSK downscaling product, which suggests a daily MODIS-derived cadence
matching the NDVI archive — in which case nothing fires. **Run
`python -m experiments.run_lst_v4 --lst-dir $LST ... --no-plots --no-stats` once
and read the Stage 0 coverage line before committing to a long run.** If coverage
is low, the choices are to gap-fill the archive first (preferred, and what keeps
the LST and NDVI results comparable) or to reduce `--memory-order` to what the
observed runs actually support.

---

## 7. Physical expectations — what a good LST run should look like

Useful for telling "the pipeline is broken" from "the field is hard", since the
NDVI reference numbers do not transfer.

**Persistence is a much weaker bar than it was for NDVI.** A daily gap-filled NDVI
field barely moves between consecutive dates (~0.007 at `t+1`). LST does: day-to-day
swings of 2–5 K from cloud, advection and a rain event are routine. So the
persistence reference printed by the error budget will be a substantially larger
*fraction* of the field's variability, and beating it should be **more** achievable
than for NDVI — the weather sensor has real work to do. If the model fails to beat
persistence on LST, suspect the emission or the operator, not the bar.

**The weather sensor should earn its place.** `fit_emission`'s held-out `R²`
gates whether the observer uses it at all. On NDVI it failed out of sample
(`R² = −198 … −948`) and was correctly dropped. On LST it should be clearly
positive — Ta and Rs are the drivers of the quantity being observed. **If it is
not, that is a finding worth chasing**, and the first things to check are that the
climatology was fitted on training only and that the anomalies are on the training
scale.

**The dominant Koopman mode should be annual**, with a stronger diurnal-residual
and weather-driven component than NDVI shows. `ρ_max = 1` remains the right choice
(v2 §3.1 calls the thermal field near-conservative, which is what motivated that
default in the first place).

**Expect a larger unpredicted-event share in the worst-pixel attribution.** An
irrigation or canal turn drops a parcel's LST by several kelvin within a day, and
nothing in a point weather record announces which parcel was watered. Those pixels
are not reducible by tuning `L`, `ρ` or the shrinkage; the honest response is the
wide predictive interval they carry. The attribution report says this explicitly.

---

## 8. Decisions taken — flag anything you disagree with

1. **One implementation, aliased entry points**, rather than a duplicated LST
   codebase. Reversible, but duplication is not.
2. **Kelvin retained**, not converted to °C (§3).
3. **Display range pinned to `[283.15, 320.15] K`**, using the span you quoted.
   This changed one existing test, which had asserted LST should auto-scale from
   its data — that was wrong on the pinning argument's own terms (§ the test's
   own docstring). If your archive exceeds this span, raise it; the panels will
   otherwise clip, which is visible rather than silent.
4. **The same span doubles as a raster validity gate**, opt out with
   `--no-physical-range`. Values outside it are masked as no-data. If your
   product is expected to produce legitimate values outside 283.15–320.15 K,
   disable this or widen the range in `dbwm/data/modality.py`.
5. **Teacher-forcing threshold 0.5 K**, chosen so it is the same *fraction* of
   the field's spatial σ that 0.02 is for NDVI (≈0.3), which keeps the reported
   "fraction of steps forced" comparable between the two runs.
6. **`inferno` for the field ramp**, `viridis` for the interval panel (they must
   stay distinguishable in a four-panel figure).
7. **Default run name becomes `dbwm_lst_gp_swiglu`**, so an LST checkpoint cannot
   overwrite an NDVI one in a shared `--ckpt-dir`. An explicit `--name` wins.
8. **The same weather CSV, the same two roles, the same season split.** No LST
   specific weather handling was added: the channels you specified
   (`Ta_C, Precip_mm, Sw_rad_mj_m2, VPD_kpa`) are exactly the ones the emission
   and forcing already consume.

Two things I could **not** verify from here and that you should confirm:

* **the LST archive's cadence and CRS** (§4, §6) — the code detects both and
  reports them, but the right response to sparse coverage is a data decision;
* **whether the LST and NDVI grids are co-registered.** They are not required to
  be — these are separate models with separate checkpoints, exactly as for v2 —
  but if you intend to difference or jointly interpret their outputs, the 136×146
  EPSG:4326 LST grid and the 135×125 EPSG:32643 NDVI grid are not pixel-coincident
  and would need a warp first.

---

## 9. Tests

```bash
JAX_PLATFORMS=cpu python -m pytest tests/ -q                  # full suite
JAX_PLATFORMS=cpu python -m pytest tests/test_lst_modality.py -q
```

`tests/test_lst_modality.py` covers the registry, the colour range and ramp, the
teacher-forcing threshold, the config resolvers, the degrees→metres conversion,
the kelvin synthetic archive, the physical gate, the raster labelling, the CV
caveat, and the alias equivalence and its refusal to run the wrong modality.
Every case pins a defect that previously ran to completion without raising.
