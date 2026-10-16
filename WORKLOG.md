# Work Log

Timestamped record of every change made while acting on [CRITICAL_REVIEW.md](CRITICAL_REVIEW.md) and the follow-up analysis.
Times are local (IST). Newest entries at the bottom.

| # | Step |
|---|---|
| 0 | Environment setup |
| 1 | Repo hygiene (.gitignore, docs folder name, requirements) |
| 2 | Config: 256 Hz, no import side effects, seeds, single source of settings |
| 3 | Raw-file parser rewrite (all pages, channel-name checks, strict numbers) |
| 4 | Stimulus timeline + per-file offset handling |
| 5 | Converter rewrite (filter full recording, block-locked epochs, metadata in NPZ) |
| 6 | Data-validity checks (fine lag scan, onset VEP, eye movements, alpha periodicity) |
| 7 | Dataset + episode sampling without replacement, block-held-out evaluation |
| 8 | Baselines (band-power LR, FBCSP-style, Riemannian tangent space) + statistics |
| 9 | Model, loss and training fixes |
| 10 | Inference fixes |
| 11 | Tests |
| 12 | Fix review's own analysis scripts |
| 13 | Docs and README |
| 14 | Full end-to-end run and results |
| — | Items that need people or hardware (handed to the user) |

---

## 2026-10-10 04:37 — Step 0: Environment
- Python 3.12.10 in `.venv`; torch 2.14.1, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1.
- 04:38 — Installed `pyriemann 0.12` (Riemannian baseline) and `pytest 9.1.1`.
- Baseline state recorded: original pipeline result in `pipeline_run.log` (mean K=5 accuracy 35.8%, invalid for the reasons in the review).

## 2026-10-10 04:39 — Step 1: Repo hygiene
- Renamed `Documenations/` → `docs/` with `git mv` (history kept). Updated links in `CRITICAL_REVIEW.md`.
- `.gitignore`: added `.venv/` (only `venv/` was listed before), `.pytest_cache/`, and generated outputs (`processed_data/`, `checkpoints/`, `results/*`).
- Untracked generated data with `git rm -r --cached processed_data` (files stay on disk; 115 CSVs, 3 NPZ, metadata.csv no longer versioned). Rebuild with `python run_pipeline.py --convert`.
- `requirements.txt`: stated Python ≥ 3.10, added `pyriemann` and `pytest`.
- Not done: moving the 96 MB of raw data to Git LFS/DVC. That rewrites git history, so it's your call (`git lfs migrate import --include="Raw data/**,Stimuli/**"`).

## 2026-10-10 04:40 — Step 2: Config
- `SAMPLING_RATE` 250 → **256 Hz**. `TRIAL_SAMPLES` is now 512 (a true 2.0 s).
- Removed the import-time `os.makedirs` side effect; added `ensure_dirs()` for entry points to call.
- Added `set_seed()` and `SEED = 42` (nothing was seeded before).
- Added block-locked epoching settings (`EPOCH_START = 0.5 s`, `EPOCH_STRIDE = 1.0 s`), channel groups (occipital, frontal pole), `BLANK_LABEL`, `OFFSETS_PATH`.
- Config is now the only source of training defaults (before: 80 epochs in config, 60 in evaluate.py, 40 in run_pipeline.py).
- Changed defaults to suit ~100s of trials: `LATENT_DIM` 128→64, `TEMPERATURE` 0.1→1.0 (logits were up to 40, saturating softmax), `DROPOUT_RATE` 0.25→0.5, `WEIGHT_DECAY` 1e-4→1e-3, `PATIENCE = 10`, `TRAIN_K_RANGE = (1, 5)`, angular loss off by default (`USE_ANGULAR_LOSS`).

## 2026-10-10 04:40 — Step 3: Raw-file parser rewrite
- New module `src/raw_io.py` (`parse_raw_txt_file`, `RawRecording`, `RawFormatError`). It replaces the old parser in `convert_raw_data.py`.
  - Parses **all pages** and joins them in time (the old parser kept page 1 only, 5.4% of the data).
  - Reads channel names from the file, normalises them (`FP2- F8` → `FP2_F8`), checks them against `CHANNELS` + `BP1/BP2` in order, and drops BP1/BP2.
  - Raises `RawFormatError` on a malformed number, missing/duplicate/reordered channel, or unequal series length. The old regex silently dropped values like `1e-3`, which shifts channels out of line.
  - Opens files strictly as UTF-8 (no `errors='ignore'`).
  - Gets the subject from the folder name (`Raw data(Subject Gope)` → `Gope`).
- Ran it on all 23 files with no errors. 69.5 min of data in total (the old parser used 3.8 min). Durations: Subject 3: 170–190 s; Gope: 180–220 s; Supra: 170–220 s.
- **Correction to the review:** a page holds **2560** samples, not 2561. The review counted the trailing comma in each series as a sample. 2560 / 256 Hz = exactly 10.0 s, so the 256 Hz conclusion is stronger. Fixed the comment in `config.py`.

## 2026-10-10 04:42 — Step 4: Stimulus timeline and offsets
- Checked the review's video timeline myself with ffmpeg: blocks of 9.2 s at 4.2 + 15·i s, 12 blocks, 5.8 s blank gaps. I tiled one frame from the middle of each block and confirmed the order ← → ↑ ↓ / L R F B / LEFT RIGHT FORWARD BACKWARD.
- The video is a WhatsApp re-encode with a slightly variable frame rate (29.98 fps average). That's fine for block timing at about 0.1 s accuracy, but get the original file for anything finer.
- New module `src/stimulus.py`:
  - `video_blocks()`: the 12 blocks with direction, format (arrow/letter/word) and repetition.
  - `stimulus_windows(duration, offset)`: windows locked to block onsets, in recording time. 7 stimulus windows per block (2 s long, starting at onset + 0.5 s, 1 s apart) and 2 blank windows per gap. Windows outside the recording are dropped.
  - `load_offsets()` / `offset_for()`: each file's offset, where recording time = video time + offset.
  - `stimulus_on_mask()`: a per-sample mask of when a stimulus is on screen.
- New file `stimulus_offsets.json`: all 23 files set to 0.0, and **marked as ASSUMED, NOT VERIFIED**. A 170 s file can't contain the whole 180 s video, so these offsets are certainly wrong for some files. The converter reads this file, so measured offsets can be added later without code changes.
- Check: a 170 s recording at offset 0 gives 77 stimulus + 22 blank windows (block 12 is cut off), and a 180 s recording gives 84 + 22.

## 2026-10-10 04:41 — Step 5: Preprocessing and converter rewrite
- `src/preprocessing.py`:
  - Filters now use second-order sections (`sosfiltfilt`) for the band-pass and the notch. They are tuned for 256 Hz and remove each channel's DC offset first.
  - New `robust_scale_recording`: median/IQR scaling over the **whole recording**, which keeps power differences between epochs. The old per-epoch `robust_standardize` is kept but no longer used by default.
  - New `euclidean_alignment` (He & Wu 2019, uses no labels) to align recordings or subjects.
  - New `flag_artifacts`: flags amplitude above 12 robust units and flat channels.
  - New `CausalPreprocessor`: the same filters run causally with carried-over state, for real-time use. `filtfilt` looks ahead and can't run online.
- `src/convert_raw_data.py` rewritten:
  - Uses the new parser, so all pages are read.
  - Filters each **full recording** before cutting epochs, so filter edge effects stay at the recording ends instead of in epoch 1 of every page.
  - Windows and labels come from the **stimulus timeline + per-file offset**. The `f_idx % 4` file-order labels are gone.
  - The NPZ now stores `X, y, is_stim, block, fmt, file_idx, start_s, bad, files, file_offsets_s, file_durations_s, subject, fs, channels`, so evaluation can hold out a block, file or format.
  - Deletes old `tensors/` and `csv_trials/` before writing, so old labels can't mix in. Per-window CSVs are now optional (`--csv`).
  - `metadata.csv` has one row per window, including offset, block, format and artifact flag.
- 04:44 — Ran it. Output: `subject_3.npz`, `subject_gope.npz`, `subject_supra.npz`.

  | Subject | Files | Stim windows | Blank windows | L/R/F/B |
  |---|---|---|---|---|
  | 3 | 12 | 959 | 266 | 252/252/252/203 |
  | Gope | 5 | 420 | 116 | 105/105/105/105 |
  | Supra | 6 | 497 | 134 | 126/126/126/119 |

  That is 1,876 labelled stimulus windows, up from 115. These counts match the review's rebuilt dataset. Backward is lower for Subject 3 because the 170 s files end before block 12.
- 04:45 — **Data-quality finding:** the artifact check flagged 0% of windows. The 99th-percentile peak is 3–4 robust units, with no difference between the frontal (FP) and other channels, meaning there are **no eye blinks** anywhere. Statistics on one file per subject:
  - The spectrum is flat from 1 to 45 Hz (power ≈ 0.105 at 1–4 Hz vs 0.084 at 30–45 Hz). Real EEG is 1/f, with delta 10–100× beta.
  - Excess kurtosis is 0.00 (Gaussian). Real EEG is heavy-tailed.
  - The correlation between consecutive samples is 0.43. For real EEG at 256 Hz it's above 0.9.
  - 24–25 distinct values per channel, std about 2.5–3 units.

  **This is low-pass filtered white noise, not EEG.** It explains why every decoder sits at chance. The review only suggested signal quality as one possibility; this makes it the leading explanation. Followed up in step 6.

## 2026-10-10 04:47 — Step 7: Dataset, episode sampling, evaluation splits, statistics
- `src/dataset.py` rewritten:
  - Loads the new NPZ metadata and keeps stimulus windows that aren't flagged as artifacts.
  - Optional `alignment='euclidean'`, applied per recording.
  - `group = recording × block` is the unit of dependence.
  - `sample_fewshot_episode`: **never samples with replacement**, and support/query always come from **different stimulus blocks**. If there isn't enough data it raises an error instead of quietly returning a ragged episode (the old code's `continue`).
  - New `calibration_splits(target, K)`: each target recording takes a turn as the calibration session. K windows per class are spread round-robin over its blocks, and the test set is **all windows from the target's other recordings**. This replaces 50 random episodes that overlapped.
- New `src/stats.py`:
  - `cluster_bootstrap_ci`: a 95% CI that resamples whole blocks.
  - `block_permutation_test`: shuffles the block → label assignment while keeping the predictions fixed. It's valid for any model.
  - `summarise` / `fmt_result`.
- Smoke test: an episode gives 20+20 windows; calibration on one recording uses 12 blocks and tests on 875 windows from other recordings, with no overlap (asserted). Random predictions score 23.5% [21.1, 25.7], p = 0.91, so the null behaves correctly.

## 2026-10-10 04:45–04:50 — Step 6: Data-validity checks (`src/sanity.py`)
- New tool `python -m src.sanity [--estimate-offsets] [--quick]`. It writes `results/sanity_report.md` and `results/sanity_lag_scan.png`. Checks:
  1. **Signal quality** per file: spectral slope (2–40 Hz), delta/beta ratio, sample-to-sample correlation, kurtosis, 50 Hz line-noise ratio, blink fraction in the FP channels, and a verdict.
  2. **Folding test with no alignment needed:** each recording is cut into 15 s cycles, and the test asks whether the cycles share a time-locked shape. The null shifts each cycle by its own random amount. Run for the occipital evoked response (0.5–12 Hz), occipital alpha power, and FP eye movements. It also works if the offsets are unknown or differ per file.
  3. **Positive control:** a synthetic block-locked occipital response is injected into real recordings, to show the folding test can detect an effect.
  4. **Fine lag scan** of stimulus-on vs blank decoding, from 0 to 15 s in 0.25 s steps (60 lags; larger lags repeat because the period is 15 s). Leave-one-file-out, balanced accuracy.
- 04:50 — Results (full run, 4 min 16 s):
  - **Signal quality: 0 of 23 files look like EEG; all 23 are "noise-like".** Spectral slope −0.03 to −0.27 (EEG is about −1 to −2.5), delta/beta 1.05–1.28 (EEG is ≫ 2), sample-to-sample correlation 0.41–0.49 (EEG is > 0.9), kurtosis about 0, almost no blink samples. Gope and Supra have **no 50 Hz line noise** (ratio 1.0–1.4), which is very unusual for a raw EEG recording. Subject 3 has some (2.7–7.7).
  - **Folding test:** 1 of 69 tests has p < 0.05, while 3.5 are expected by chance. **None survive Bonferroni correction.** There's no block-locked evoked response, alpha change or eye movement in any file.
  - **Positive control:** an injected response of 0.1× channel SD is found in 0/6 files, 0.2× in 0/6, 0.3× in 4/6, and 0.5× in 6/6. So the test can detect a modest real effect, and its negative result means something.
  - **Lag scan:** Subject 3 is best at 51.9% (mean 50.1%), Gope 55.4% (mean 51.4%), Supra 56.5% (mean 52.0%). The best of 60 noisy scores is expected to reach about this level by chance. No offset works.
  - `--estimate-offsets`: no file has a significant response, so no offsets can be estimated. Changed the code to skip writing an empty `stimulus_offsets.estimated.json`.
- **Conclusion:** the recordings don't contain detectable EEG. The likely causes are the export (wrong channel/gain/units, or already filtered), electrodes not connected or not making contact, or synthetic/placeholder data. Synchronisation alone can't explain this, because the folding test doesn't depend on it. **No decoder can learn direction from these files.** The remaining steps still fix the code so it will be correct once valid data exists.

## 2026-10-10 04:56 — Step 9: Model, loss and training fixes
- `src/models.py`:
  - The spatial convolution in `SpatialTemporalProtoNet` is now **really depthwise** (`groups=F1`, depth multiplier 2). The docstring called it depthwise but it was a full 32→64 convolution.
  - Added a 0.5 s temporal kernel (128 samples) for slower components, next to the 0.25 s and 0.125 s kernels.
  - The separable convolution is now depthwise + pointwise, as the name says.
  - Removed the redundant LayerNorm before L2 normalisation, and the docstring's promise of a `RiemannianMDMClassifier` that never existed (the Riemannian methods are in `baselines.py`).
  - **Parameters: 112,832 → 7,280.** The EEGNet backbone is 6,576.
  - New `build_model`, `count_parameters`, and `adapt_batchnorm` (AdaBN: re-estimates BatchNorm statistics on the target subject's unlabelled data).
- `src/losses.py`:
  - Fixed the angular-loss docstring, which contradicted the target matrix.
  - Marked that loss as an untested bias, now off by default.
  - Default temperature 1.0, since squared distances are ≤ 4.
  - New `compute_prototypes(z, labels)`, which uses the labels instead of assuming a fixed order.
- `src/train_fewshot.py` rewritten:
  - Fixed the **missing `import glob`**, which crashed README step 2.
  - Seeds set.
  - **K drawn per episode from 1–5**; before, it always trained at K=5 and tested at 1/3/5.
  - **Early stopping on validation loss from held-out recordings** (one per training subject). The old code never validated, so it printed "Val Acc 0.00%" every epoch.
  - The best model is restored, and the **checkpoint is always saved** with the full config and history.
  - Augmentation: amplitude scaling, noise, channel dropout, ±125 ms shift.
  - Support and query go through the network in one batch, so BatchNorm sees both.
  - Output is flushed, so progress shows up when piped.
- `src/evaluate.py` rewritten:
  - LOSO with Euclidean alignment and AdaBN on the target.
  - Calibration splits from step 7 (calibrate on one recording, test on the others).
  - A block-bootstrap CI and block-permutation p-value for every number.
  - A **shuffled-label control model trained in every fold**.
  - ITR is reported only when p < 0.05.
  - Saves a checkpoint per fold, writes `results/loso_<tag>.json` and `.md`, and plots the K curve with CIs plus the control and confusion matrix.
  - Default K is 1, 3, 5, 10. K=10 is now possible without duplicates, since each recording holds 21 windows per class.
- 04:56 — Smoke test (2 epochs, K=1 and 5) ran end to end. Every number was 23.9–25.9% with p ≥ 0.32, and real labels were indistinguishable from the shuffled control. Smoke-test outputs deleted.

## 2026-10-10 04:57 — Step 10: Inference fixes (`src/inference.py`)
- `calibrate()` **now uses `support_labels`**: ids or names, any order, any number per class, with an error if a direction is missing. Before, it ignored the labels and assumed `[L×K, R×K, F×K, B×K]` order.
- The model type, latent size, dropout, channel count and **temperature are read from the checkpoint's `config`**. Before, 128 and 0.1 were hard-coded.
- `torch.load(..., weights_only=True)`. A missing checkpoint now raises an error instead of silently using random weights.
- Optional AdaBN during calibration (`adapt_bn_on=` unlabelled windows from the new subject).
- `predict_from_csv` checks the channel count.
- New `StreamingPredictor` for real-time use: `CausalPreprocessor` fitted on a calibration recording, then chunk-by-chunk causal filtering and classification of the latest 2 s. The old "real-time" claim was impossible with `filtfilt`, which needs future samples.
- New `align_windows` helper, matching the Euclidean alignment used in training.
- Smoke test: trained a 1-epoch model, saved and reloaded it, calibrated on 40 windows in random order with string labels (6/11/14/9 per class), predicted one window, and streamed 4 s of raw data in 64-sample chunks. All worked. Temporary checkpoint deleted.

## 2026-10-10 04:52–05:10 — Step 8: Classical baselines + statistics (`src/baselines.py`)
- New tool `python -m src.baselines [--quick]`, writing `results/baselines.md` and `results/baselines.json`.
- Four pipelines:
  - log band power (6 bands × 18 channels) + LR;
  - Riemannian tangent space + LR;
  - filter-bank multiclass CSP (4 bands) + LR;
  - Riemannian MDM.
- Schemes:
  - within-subject leave-one-file-out;
  - **true leave-one-format-out** (`groups = format`);
  - a **shuffled-label control** (block labels shuffled within each recording);
  - cross-subject LOSO with Euclidean alignment;
  - a K-shot nearest-class-mean on band power, using the **same calibration splits as the deep model** (K = 1, 3, 5, 10).
- Every number comes with a block-bootstrap 95% CI and a block-permutation p-value.
- Moved `shuffled_block_labels` into `dataset.py` so the evaluation can use it too.
- Results (full table in `results/baselines.md`):
  - Within-subject: 19.0–29.8% across all pipelines, subjects and schemes.
  - Cross-subject: 22.8–27.9%.
  - Few-shot NCM: 23.9–26.1% at every K.
  - **1 of 48 tests with real labels has p < 0.05** (Gope, tangent space, leave-one-file-out, p = 0.046), where 2.4 are expected by chance. **2 of 12 shuffled-label controls also have p < 0.05** (p = 0.043 and 0.021). That is exactly the noise floor: no baseline can decode direction from these data.

## 2026-10-10 05:02 — Step 11: Tests (`tests/`, 37 tests, run with `python -m pytest tests`)
- `conftest.py`: writes synthetic raw files in the exact export layout and synthetic NPZ files in the converter's format, optionally with a class-dependent signal.
- `test_raw_io.py`: all pages parsed in order, aux channels dropped, malformed numbers / reordered or missing channels raise errors, name helpers.
- `test_stimulus.py`: block order and formats, windows inside blocks/gaps, 170 s truncation, offsets shift windows, nothing before t = 0, the on-mask, offset lookup.
- `test_dataset.py`: episodes **never repeat a window**, support/query use **disjoint blocks**, labels match; not enough data raises an error; calibration splits are cross-recording with exactly K per class; shuffled labels stay constant within a block; Euclidean alignment whitens each recording.
- `test_signal_and_stats.py`:
  - the notch removes 50 Hz at 256 Hz;
  - **the causal preprocessor doesn't use future samples**;
  - artifact flags work;
  - the signal-quality check rejects white noise and accepts EEG-like pink noise with blinks;
  - the folding test detects a periodic response but not noise;
  - permutation test and CI behave correctly.
- `test_models_inference.py`:
  - embeddings are unit-norm;
  - the spatial conv is depthwise;
  - AdaBN updates statistics;
  - prototypes use labels;
  - **calibration doesn't depend on order**, and a missing direction raises an error;
  - **end to end, the pipeline learns a real synthetic signal** (accuracy > 50%, p < 0.01). So chance results on the real data aren't caused by a broken pipeline.
- **Bug found by the tests:** in `sanity.signal_quality`, adding numpy booleans with `+` is a logical OR, so the "EEG-like score" could never exceed 1. Fixed by counting explicitly. Re-checked: every real file scores 0 of 4 criteria, so the step 6 verdicts don't change.
- 05:02 — **37 passed** (73 s).

## 2026-10-10 05:04 — Step 12: Fixes to the review's own analysis
- `analysis/diagnostics.py`: the "leave-one-format-out" test grouped by (file, format) with `GroupKFold(3)`, which mixes formats in every fold. Fixed to `LeaveOneGroupOut` on format.
  - Reran it (`results/diagnostics_rerun.log`). Corrected values: Subject 3 **23.2%** (review said 26.9%), Gope **22.1%** (same), Supra **23.9%** (review said 20.5%). All other numbers reproduce the review exactly.
- `analysis/stim_on_vs_blank.py`: removed the `exec()` of part of another file's source. It now imports `src.sanity` and scans 60 offsets (0–15 s in 0.25 s steps) instead of 6 lags that were only 3 distinct offsets.
- Added **§9 Errata and follow-up** to `CRITICAL_REVIEW.md`: 2560-sample pages, the format-out fix, the lag-scan periodicity, files shorter than the video, the overstated stim-vs-blank expectation, and the new finding that the recordings aren't EEG-like.

## 2026-10-10 05:04 — Step 13: Docs and README
- `README.md` rewritten:
  - a status section stating plainly that the data has no detectable EEG and the earlier accuracies were artifacts;
  - the correct structure (no `e:\EEG-direction-detection\` path, no garbled emoji headings);
  - commands for sanity / convert / baselines / evaluate / tests;
  - an inference example using the label-aware `calibrate` and `StreamingPredictor`, with a checkpoint the pipeline actually produces;
  - the evaluation protocol.
- `docs/DATA_CONVERSION_AND_EXECUTION_GUIDE.md` rewritten to match the code: raw format (2560-sample pages at 256 Hz), stimulus-timeline labels and the offsets file, preprocessing order, NPZ fields, commands, evaluation rules, **what the recording side must provide** (export check, eyes-open/closed check, sync, original video, more subjects), and troubleshooting. Removed the pasted copy of the old converter.
- `docs/RESEARCH_DOCUMENTATION.md`: added an **implementation status table** at the top, mapping each claim to reality. The Conformer/attention, MAML, self-supervised pre-training and ICA/ASR aren't implemented; the Riemannian methods exist only as baselines. Fixed 250 Hz → 256 Hz, 500 → 512 samples, and the epoching and normalisation descriptions.
- `run_pipeline.py` rewritten. It runs sanity → conversion (if needed) → baselines → LOSO with control. Subprocess output is unbuffered, and the default K is 1, 3, 5, 10.

## 2026-10-10 05:20 — Step 14 (attempt 1): Full pipeline run, interrupted
- Ran `python run_pipeline.py --convert` (log: `results/pipeline_run.log`).
  - Sanity checks, conversion and baselines finished, with the same results as steps 5, 6 and 8 (seeded; e.g. Gope tangent-space leave-one-file-out = 29.8% again).
  - The LOSO step stopped at fold 1, epoch 10, when the previous session ended.
- Partial LOSO: validation loss was **1.3865 = ln 4** from epoch 1 to 10, the loss of a model that predicts every class equally. The network learned nothing that transfers to held-out recordings, which is consistent with the data findings.

## 2026-10-10 19:55 — Step 14 (attempt 2): LOSO only
- Restarted just the LOSO evaluation in the background: `python -m src.evaluate` (60 epochs max with early stopping, K = 1/3/5/10, shuffled-label control in each fold). Log: `results/loso_run.log`.

## 2026-10-10 20:27 — Acquisition system (from the user)
- Device: **Medicaid Systems NeuroMax** EEG amplifier, recorded with the NeuroMax's own acquisition software (not Elora). The device records 0.5–100 Hz; published NeuroMax studies report 256 Hz, which matches the files.
- Hypothesis (unconfirmed): `AllPages_*.txt` is a page/screen export from review mode, not raw data. Evidence: 10 s pages, the bipolar display montage, whole-number values with ~24 levels, almost no power above ~50 Hz, and a 50 Hz notch on for Gope and Supra but off for Subject 3. A display export would include display filters and sensitivity scaling.
- Requested from the user: how the text files were exported; whether the software has an EDF/EDF+ or µV raw export; whether it can mark events (video start) during recording. Plan: re-export one existing session in raw form and run `src.sanity` on it before recording anything new.

## 2026-10-10 20:33 — Step 14 (attempt 2): LOSO results, complete
- `python -m src.evaluate` finished with exit code 0. Outputs:
  - `results/loso_spatial_temporal_euclidean_adabn.{md,json}`
  - the K curve and confusion-matrix plots
  - one checkpoint per fold in `checkpoints/`
- Early stopping: fold 1 at epoch 46 (best 36), fold 2 at epoch 38 (best 28), fold 3 at epoch 14 (best 4). Best validation loss was 1.3863–1.3866 every time, which is **ln 4 = 1.3863, the loss of guessing evenly**.

  | Target | Run | K=1 | K=3 | K=5 | K=10 |
  |---|---|---|---|---|---|
  | 3 | real | 25.2% | 25.3% | 24.5% | 24.6% |
  | 3 | shuffled control | 25.1% | 25.1% | 24.4% | 24.3% |
  | Gope | real | 25.1% | 24.6% | 27.2% (p = 0.084) | 25.8% |
  | Gope | shuffled control | 24.3% | 26.2% | 27.1% | 26.4% |
  | Supra | real | 25.1% | 25.2% | 25.1% | 24.2% |
  | Supra | shuffled control | 24.1% | 25.9% | 25.7% | 24.7% |

- **No result is significant (all p ≥ 0.08), and real labels are indistinguishable from the shuffled control.** Accuracy doesn't rise with K. Compared with the original pipeline's 33–36% mean (Gope 55%), the difference is entirely the artifacts the review identified.
- Renamed the original pipeline's plots to `results/OLD_INVALID_pipeline_*.png` so they can't be mistaken for current results.

## Status at the end of this pass
- **Code:** every item in the review's P1–P3 has been done. P0 item 4 (parser) is done. 37 tests pass. The full pipeline (`run_pipeline.py`) runs end to end.
- **Data:** all evidence (signal quality, alignment-free folding test, 60-offset lag scan, 4 baselines × 5 schemes, ProtoNet LOSO with controls) agrees that **the current text exports contain no decodable stimulus information and don't look like EEG**.
- **Handed to the user (needs people or hardware):**
  1. Find out how `AllPages_*.txt` was exported from the NeuroMax software. Get a raw export (EDF/EDF+ or µV, referential montage, filters off, full resolution) of one existing session and run `python -m src.sanity` on it.
  2. Record an eyes-open/eyes-closed check.
  3. Mark the video start (event marker or photodiode) in every session, or record each file's offset in `stimulus_offsets.json`.
  4. Get the original stimulus video.
  5. Add more subjects.
  6. Decide on committing these changes, and on moving the raw data to Git LFS.

## 2026-10-10 21:12 — EDF export of the text files
- The user asked whether a snippet (`np.loadtxt` → MNE `RawArray` → `export`) could convert the files. It can't as written: the files are one tab-separated line of 10 s pages with 20 named series, not a comma table, and the snippet's 6 channel names don't match the 18 bipolar channels.
- New `src/export_edf.py` (`python -m src.export_edf [--out DIR] [--keep-aux] [--no-annotations]`). It reuses `raw_io`, so all pages are read and channel names checked. It writes 18 bipolar channels named `FP2-F8`… at 256 Hz, and adds **stimulus-block annotations** ("left / arrow", …, plus "video start"), tagged "(offset assumed)" because the offsets aren't verified. Values are written as µV, but their real unit and gain are **unconfirmed**.
- Installed `mne 1.13.2` and `edfio 0.4.18`; added both to `requirements.txt`.
- Converted all 23 files to `processed_data/edf/` (38 MB). Read-back check: 256 Hz, 18 channels, correct duration, max difference 0.024 (EDF 16-bit rounding), annotations present. New test `tests/test_export_edf.py` passes.
- **Caveat:** EDF is only a container. These EDFs hold the same noise-like signal; converting doesn't repair it. Their use is to open the data in a standard viewer (e.g. EDFbrowser) and see it directly. A **raw** export from the NeuroMax software is still needed.

## 2026-10-10 21:29 — Full data audit against a real EEG reference
- The user asked to recheck all the data and explain what's wrong, why, and what it should look like.
- Downloaded a reference recording to the scratchpad: PhysioNet EEGMMIDB subject 1, runs 1 (eyes open) and 2 (eyes closed). Converted it to the same 18 bipolar channels (T3/T4/T5/T6 = T7/T8/P7/P8) at 256 Hz.
- New `analysis/data_audit.py`. It writes 5 figures and `audit_stats.json` to `docs/figures/data_audit/`. Colours were checked with the dataviz palette validator.
- New checks on top of earlier ones:
  - **The channels add up correctly:** the two right-side chains FP2→O2 sum identically (r = 1.000), and the same on the left. The files are genuine bipolar derivations of 21 inputs.
  - **Neighbouring-channel correlation −0.49** (independent noise per electrode predicts −0.5; real EEG gives +0.33). Channels far apart correlate about 0.
  - **DC offsets are identical across all 3 people and 23 sessions** (e.g. FP1-F7 2034 ± 1.8, F4-C4 −1080 to −1110). These are the amplifier's own offsets, not electrode contact. This is the strongest single clue.
  - Signal size barely changes over time (6% variation between 2 s windows, vs 28% in real EEG). No duplicated pages (0 of 417).
- Your files vs real EEG eyes open / eyes closed:
  - spectral slope −0.11 vs −1.17 / −1.15;
  - slow/fast power 1.2 vs 33 / 9;
  - occipital alpha peak 1.2× vs 2.9× / 39×;
  - frontal kurtosis 0.0 vs 16 / 0.6;
  - neighbouring-channel correlation −0.49 vs +0.33;
  - distinct values 25 vs 15,616.
- The only file with any change: Subject 3 `AllPages_TEST.txt`, pages 6–9 (about 50–90 s). A narrow 12.75 Hz peak and spikiness appear only in the left parasagittal chain (FP1-F3, F3-C3, C3-P3, P3-O1). That looks like one loose electrode or electrical pickup, not brain activity.
- Wrote `DATA_AUDIT.md`, a plain-language report with the figures. It covers what's fine, what's wrong and why with the expected appearance, ranked likely causes (electrodes not delivering signal > wrong export > demo data), and an action table.
