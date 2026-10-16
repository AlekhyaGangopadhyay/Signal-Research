# Critical Review: Few-Shot EEG Direction Detection

Review date: 2026-10-10. Scope: every file in `src/`, `run_pipeline.py`, the raw data, the processed data, the stimulus video and the pipeline results.
The diagnostic experiments behind this report are in [analysis/](analysis/) and can be re-run with
`.venv/Scripts/python -I analysis/diagnostics.py .`

---

## 1. Bottom line

**The reported results do not measure direction decoding, and the pipeline as written cannot.**
The labels are not derived from the experiment. Each raw file is a single 3-minute session in which the subject saw **all four directions**, but the converter labels a whole file as one direction based on its position in an alphabetical sort. On top of that, the converter reads only the first 10 seconds of each file, the evaluation re-uses calibration trials as test trials, and the sampling rate is wrong.

When I rebuilt the dataset with labels taken from the stimulus video, a standard band-power + logistic-regression baseline scored at chance (20–28%) for every subject and every validation scheme. Even "arrow on screen vs. blank screen", which should be the easiest contrast available, was at chance (43–57%). So either the EEG files are not time-aligned to the video, or the recordings carry no usable stimulus signal at this quality. That has to be resolved before any modelling work matters.

| Issue | Severity | Effect |
|---|---|---|
| Labels assigned by file order, not by stimulus | **Critical** | All reported accuracies are meaningless |
| No event markers / sync between EEG and video | **Critical** | Correct labels cannot currently be recovered with confidence |
| Only page 1 (10 s) of each ~180 s file is used | **Critical** | 94.6% of the data is discarded |
| Support/query overlap from sampling with replacement | High | Inflates Gope and Supra scores (up to 50% of test trials are copies) |
| Sampling rate is 256 Hz, not 250 Hz | Medium | Filters and the notch are mis-tuned; epochs are 1.95 s, not 2 s |
| Validation is never run (`val_dataset=None`) | Medium | "Val Acc 0.00%" is printed every epoch; no checkpoint is ever saved |
| Severe overfitting (train loss ≈ 1e-4 by epoch 10) | Medium | Model memorises ~85 trials |
| `inference.py` ignores `support_labels` | Medium | Calibration silently assumes a fixed order |
| Docs/README describe things the code does not do | Low | Misleading for readers and reviewers |

---

## 2. The pipeline run

Run with the defaults (`python run_pipeline.py`: 40 epochs, K ∈ {1,3,5}), on CPU, about 45 min. Log: [pipeline_run.log](pipeline_run.log).

| Held-out subject | K=1 | K=3 | K=5 | κ at K=5 |
|---|---|---|---|---|
| Subject 3 | 22.9% | 24.1% | 25.3% | 0.004 |
| Gope | 46.0% | 51.7% | 55.4% | 0.405 |
| Supra | 30.1% | 28.6% | 26.6% | 0.021 |
| **Mean** | **33.0%** | **34.8%** | **35.8%** | |

Chance is 25%.

**What the numbers actually show**

- **Subject 3 is exactly at chance.** It is the only subject with enough trials per class (15) to avoid duplicate sampling (see §4.2), which makes it the only clean evaluation, and it shows nothing.
- **Gope's 55% is an artefact.** At K=5, half of Gope's query trials are byte-identical copies of support trials. A copy sits at distance 0 from its own prototype, so it is classified correctly by construction. Gope also has the largest gain with K, which is the pattern you'd expect from more duplicates, not from more information.
- **The "Forward" diagonal (55%) in the confusion matrix comes from the same artefact.** Forward has only 5 trials for Gope and Supra, so it always needs replacement sampling.
- **Training.** Train accuracy reaches 100% and train loss reaches ~1e-4 within 10 epochs, then stays flat for 30 more epochs. The model is memorising.
- **"Val Acc 0.00%" is not a measurement.** `evaluate.py` passes `val_dataset=None`, so validation is skipped, and the 0 is just the initial value.

---

## 3. The data and the experiment (what the code assumes vs. what is there)

### 3.1 Raw file format
Each `.txt` file is a single line of tab-separated fields. The file is split into pages (`Page-1`, `Page-2`, …). Each page has **20** labelled series of **2561** samples: the 18 bipolar EEG channels plus `BP1-REF` and `BP2-REF`, which are large constant offsets (≈ −4280 and −5220) and look like unused auxiliary inputs. Values are integers with very coarse quantisation: channel 0 of Subject 3 has only 24 distinct values and a std of ~2.8 units, which suggests a low-gain export.

### 3.2 Sampling rate is 256 Hz
- One page is 2561 samples. At 256 Hz that is exactly 10.0 s; at 250 Hz it would be 10.24 s.
- Under fs=256 the power-line peak lands at **50.0 Hz**. Under the assumed fs=250 it lands at **48.7–49.2 Hz**.
- File lengths come out as exactly 170, 180, 190 or 220 s at 256 Hz.

Consequences: the notch filter is at the wrong frequency (it is designed for 50 Hz at 250 Hz, which corresponds to 51.2 Hz in real time), all band edges are 2.4% off, and the "2.0 s" epochs are 1.95 s.

### 3.3 The stimulus protocol: every file contains every direction
The video in `Stimuli/` is 180.2 s long. Extracting its frames gives a clean timeline:

- a title card ("visual stimuli") at 0.5–1.5 s;
- **12 stimulus blocks of 9.2 s**, with onsets at 4.2 s + 15 s·i (i = 0…11), separated by 5.8 s blank screens;
- block order: ← → ↑ ↓ (arrows), then L R F B (letters), then LEFT RIGHT FORWARD BACKWARD (words).

So **one 180 s recording = one viewing of this video = 3 presentations of each of the 4 directions.** The recording lengths (170–220 s) match this one-file-per-viewing picture.

### 3.4 How the converter labels data ([src/convert_raw_data.py:105](src/convert_raw_data.py#L105))
```python
dir_id = f_idx % len(DIRECTIONS)
```
- The label is the file's index in an **alphabetical** sort, modulo 4. For Subject 3 the order is `1, 10, 11, 2, 3, …, 9, TEST`, so file 10 is "right", file 2 is "backward", and `AllPages_TEST.txt` is training data labelled "backward".
- Since every file contains all four directions, any file-level label is wrong for about 75% of its content, whatever order is used.
- The data guide itself says the files "do not contain markers" ([docs/DATA_CONVERSION_AND_EXECUTION_GUIDE.md:38](docs/DATA_CONVERSION_AND_EXECUTION_GUIDE.md#L38)). The cyclic assignment is a placeholder that was never replaced.

### 3.5 Only page 1 is parsed ([src/convert_raw_data.py:54](src/convert_raw_data.py#L54))
`channels_data[:NUM_CHANNELS]` keeps the first 18 series in the file, which are page 1's 18 channels. The other 16–21 pages are thrown away. The result is 5 epochs per file, 115 epochs in total, from 69.5 min of recording: **5.4% of the data**. Those first 10 s are also mostly the title card and the blank lead-in, before the first arrow appears at 4.2 s.

### 3.6 The class imbalance comes from the file counts

| Subject | Files | Epochs | Per class (L/R/F/B) |
|---|---|---|---|
| Subject 3 | 12 | 60 | 15/15/15/15 |
| Gope | 5 | 25 | 10/5/5/5 |
| Supra | 6 | 30 | 10/10/5/5 |

---

## 4. Evidence from diagnostic experiments

All of these use log band-power features (6 bands × 18 channels) with a regularised logistic regression. That is a standard, strong baseline for this amount of data.

### 4.1 The current labels are file identity
Using exactly the pipeline's epochs and labels:

| Subject | Decode *file ID* (chance) | "Direction", random split | "Direction", leave-one-file-out |
|---|---|---|---|
| Subject 3 | 23.3% (8%) | 36.7% | **8.3%** |
| Gope | 36.0% (20%) | 48.0% | **12.0%** |
| Supra | 66.7% (17%) | 56.7% | **10.0%** |

Each recording has a recognisable fingerprint (electrode impedance, noise, state), and since label = file, a classifier can look accurate under a random split by recognising the file. Once a whole file is held out, accuracy drops **below** chance. This is the classic session-confound signature. Any within-subject result on these labels would be inflated in the same way.

### 4.2 Calibration (support) and test (query) trials overlap ([src/dataset.py:61](src/dataset.py#L61))
When a class has fewer than K+Q trials, `sample_fewshot_episode` samples **with replacement**, and support and query are drawn from the same pool. Simulated with the real class counts:

| Subject | K=1 | K=3 | K=5 |
|---|---|---|---|
| Subject 3 | 0% | 0% | 0% |
| Gope | 15% | 37% | **50%** |
| Supra | 10% | 24% | 34% |

This is the percentage of query trials that are exact copies of a support trial.

### 4.3 Re-labelled with the video timeline
I re-epoched **all pages** at 256 Hz, using 2 s windows starting 0.5 s after each block onset (1 s stride, 7 windows per block), labelled by the video, and assumed the recording starts at video t = 0.

| Subject | Epochs | Leave-one-file-out | Leave-one-format-out | Shuffled-label control |
|---|---|---|---|---|
| Subject 3 | 959 | 24.0% | 26.9% | 29.2% |
| Gope | 420 | 19.8% | 22.1% | 23.3% |
| Supra | 497 | 21.5% | 20.5% | 24.9% |

Cross-subject, training on 2 subjects and testing on the third: 25.2% / 27.6% / 25.8%.

**Every result is at chance and indistinguishable from the shuffled control.**

### 4.4 Sanity check: stimulus on vs. blank screen
Each 2 s window is labelled "stimulus on" or "blank screen", and the alignment is shifted from −5 s to +40 s:

| Subject | −5 s | 0 s | +5 s | +10 s | +20 s | +40 s |
|---|---|---|---|---|---|---|
| Subject 3 | 53% | 53% | 48% | 51% | 45% | 53% |
| Gope | 49% | 57% | 53% | 53% | 51% | 52% |
| Supra | 49% | 51% | 43% | 49% | 47% | 47% |

Chance is 50%. A high-contrast black arrow appearing on a white screen normally produces clear occipital changes (visual evoked responses and alpha suppression). Not detecting them at any shift means one of these is true:

- the recordings are **not synchronised** with the video, or the offset is outside this range or varies by file;
- the signal quality is too low (see the quantisation in §3.1);
- the subjects were doing something other than watching this video during these files.

**This is the first thing to resolve.** No model can recover direction from data that does not separate "something on screen" from "nothing on screen".

---

## 5. Code review, file by file

### `src/convert_raw_data.py`
- **[Critical]** File-level labels by alphabetical index (§3.4).
- **[Critical]** Only the first 18 series are used, i.e. page 1 only (§3.5). Pages should be grouped in blocks of 20, the 18 EEG series kept from each, and the pages concatenated in time.
- The regex `^-?\d+(\.\d+)?$` silently drops values such as `1e-3` and `+5`, and dropping values misaligns the time axis across channels. A malformed sample should raise an error rather than be skipped.
- Channels are identified by position, not by name. The labels (`FP2- F8`, …) are in the file and should be parsed and checked against `CHANNELS`.
- The `AllPages_TEST.txt` file is treated as ordinary training data.
- Non-overlapping epochs starting at sample 0 ignore stimulus timing completely. The docs promise "[0.0 s, 2.0 s] post stimulus onset".
- Writing one CSV per trial (115 files, 21 MB) is redundant with the NPZ. It should be optional.

### `src/preprocessing.py`
- Uses `filtfilt` with `(b, a)` for a 4th-order band-pass that goes down to 0.5 Hz. Second-order sections (`butter(..., output='sos')` + `sosfiltfilt`) are numerically safer at low cut-offs.
- `robust_standardize` normalises each epoch and channel separately. That removes absolute power differences between conditions, which are often the most informative feature in visual paradigms. Normalising per recording or per subject (or using Euclidean/Riemannian alignment) would keep that information while still handling subject differences.
- There is no artefact handling: no rejection of eye blinks (frontal FP channels), muscle activity or flat or saturated channels.

### `src/config.py`
- `SAMPLING_RATE = 250.0` is wrong (§3.2); it should be 256.
- The module creates directories as a side effect of being imported.
- There are no random seeds anywhere, so runs are not reproducible.

### `src/dataset.py`
- **[High]** Replacement sampling and support/query overlap (§4.2). For evaluation, sample K support trials without replacement and evaluate on **all remaining** trials of the target subject.
- If a class is missing, the `continue` produces a ragged episode, and later `.view(4, k, -1)` calls then crash or mix classes up.
- Episodes ignore where trials came from. Support and query should come from **different recordings or blocks**, so that file fingerprints can't be used.

### `src/models.py`
- `SpatialTemporalProtoNet` is described as "depthwise", but `conv_spatial` is a full 32→64 convolution over all 18 channels (no `groups`), with about 37k parameters in that layer alone. That is a lot of capacity for 85 training trials.
- `LayerNorm` followed by L2-normalisation is redundant.
- The docstring promises a `RiemannianMDMClassifier` that does not exist. `EEGNetBackbone` is never used by the pipeline.
- The multi-scale branch only has kernels of 32 and 64 samples (125–250 ms). Nothing captures the slow components of visual responses, and there is no attention despite the "Separable Attention" comment.

### `src/losses.py`
- `DirectionalAngularMarginLoss` forces the prototypes into a geometry where left is opposite right and forward is opposite backward. That assumes the brain codes these directions as geometric opposites, and nothing supports that for arrows, letters and words. It is an untested inductive bias and needs an ablation.
- The comments contradict each other (one says "Left vs Forward orthogonal"; the matrix and code are consistent but the docstring angles differ).
- With L2-normalised embeddings, squared distance is at most 4. Temperature 0.1 gives logits up to 40, which pushes the softmax into saturation and helps explain the 1e-4 training loss.

### `src/train_fewshot.py`
- `glob` is used in `__main__` but **never imported**, so `python -m src.train_fewshot` (the README's step 2) crashes with a `NameError`.
- 100 episodes × 40 epochs over about 85 trials, with no early stopping, means the same trials are seen thousands of times.
- The best-checkpoint logic only runs when `val_dataset` is given, which LOSO never does. With `save_path` also `None`, the LOSO pipeline saves no model, and the `checkpoints/` folder stays empty.
- There is no data augmentation (time shifts, amplitude scaling, channel dropout, mixup), which is the usual remedy at this data size.

### `src/evaluate.py`
- `n_eval_episodes=50` random episodes are averaged. That gives the mean of a resampled, overlapping estimate, not a held-out accuracy (§4.2).
- No confidence intervals or permutation tests are reported. With 3 subjects, any average needs a significance test against chance.
- Kappa is computed on pooled episodes that contain duplicates, and ITR is computed from the inflated accuracy.
- The function default is `k_shots=[1,3,5,10]`, but K=10 is impossible for Gope and Supra without even more duplication.

### `src/inference.py`
- `calibrate()` **ignores `support_labels`**. It assumes the trials are ordered `[L×K, R×K, F×K, B×K]`, and any other order silently gives wrong prototypes.
- It hard-codes temperature 0.1 and `latent_dim=128` instead of reading them from the checkpoint's `config`.
- The README example loads `checkpoints/protonet_spatial_temporal_5shot.pth`, which no part of the pipeline produces.

### `run_pipeline.py` and repo hygiene
- The 45-min run prints nothing until it finishes when output is piped. It should use `print(..., flush=True)` or `python -u`, plus `tqdm` for progress.
- `requirements.txt` has no Python version. It fails on Python 3.7 (torch ≥ 2 and pandas ≥ 2 need ≥ 3.8/3.9).
- Generated data (21 MB of CSVs and the NPZ files) is committed to git.
- The folder name `Documenations` is misspelt.
- The README shows a different machine's path (`e:\EEG-direction-detection\`) and has mojibake (garbled characters) in its headings.

---

## 6. Documentation vs. reality
- RESEARCH_DOCUMENTATION.md describes a "Riemannian few-shot representation", a Riemannian MDM baseline and attention modules. None of these exist in the code.
- The data guide documents file-level cyclic labelling as the method, while also noting that the files have no markers.
- The README says evaluation covers K = 1, 3, 5, 10. The pipeline runs 1, 3, 5.

---

## 7. Recommendations, in priority order

### P0: make the data valid (nothing else matters until this is done)
1. **Establish synchronisation.** Find out from whoever ran the sessions how recording start relates to video start. Better, re-record with a hardware or software trigger (a photodiode on the screen corner, or an event marker sent to the amplifier at each block onset). If the amplifier software can export annotations, export them.
2. **Verify signal quality.** Check the export gain and units (24 distinct values per channel is far too coarse for EEG). Re-export in µV with full resolution if the acquisition software allows it.
3. **Validate with a known effect before trying direction.** Show that "stimulus vs. blank" or occipital alpha blocking is decodable above chance. If that fails, the data cannot support the direction question.
4. **Fix the parser.** Use all pages, check channel names, drop BP1/BP2, set fs=256, and epoch relative to block onsets with labels taken from the stimulus timeline in [analysis/stimulus_timeline.py](analysis/stimulus_timeline.py).

### P1: make the evaluation honest
5. Split by **recording or block** inside every subject, never by random epoch. For LOSO, take K support trials per class from **one block** of the target subject and test on the target's **other** blocks.
6. Remove replacement sampling. Report accuracy with a 95% CI and a permutation-test p-value, plus a shuffled-label control, for every result.
7. Always run simple baselines next to the deep model: band-power + LDA/LR, CSP/FBCSP, and Riemannian tangent-space + LR (`pyriemann`). If the deep model can't beat them, it shouldn't be the headline.
8. Separate the three presentation formats (arrow, letter, word). Decoding "arrow-left vs arrow-right" is a visual-shape problem; generalising across formats is the real "direction" claim. Test it explicitly with leave-one-format-out.

### P2: modelling (only once P0 and P1 show a signal)
9. Use a smaller model or a real depthwise spatial convolution (EEGNet-style, `groups=F1`), plus early stopping on a held-out block, augmentation, and seeds.
10. Normalise per subject or recording, or align with Euclidean or Riemannian alignment, instead of z-scoring each epoch.
11. Ablate the angular-margin loss, and keep it only if it measurably helps.
12. Pre-train on public datasets, such as visual-cue motor-imagery corpora (BCI IV-2a) or VEP data, so the meta-learner sees many subjects rather than 2.
13. Collect more subjects. LOSO with 3 subjects means each model is trained on 2, which is too few for any statement about cross-subject generalisation.

### P3: engineering
14. Fix `import glob` in train_fewshot, the `support_labels` handling in inference, checkpoint saving in LOSO, unbuffered output, and seeds.
15. Pin the Python version, stop committing generated data, and bring the README and docs in line with what the code actually does.

---

## 8. Reproducing this review
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Python ≥ 3.9
.venv/Scripts/python run_pipeline.py                     # original pipeline (≈45 min CPU)
.venv/Scripts/python -I analysis/diagnostics.py .        # §3–§4 tables
.venv/Scripts/python -I analysis/stim_on_vs_blank.py .   # §4.4
.venv/Scripts/python -I analysis/stimulus_timeline.py "Stimuli/WhatsApp Video 2026-10-09 at 11.10.30 PM.mp4"  # needs ffmpeg
```

---

## 9. Errata and follow-up (2026-10-10, after review)

Corrections to this review, found while acting on it (details in [WORKLOG.md](WORKLOG.md)):

1. **Page length.** A page holds **2560** samples, not 2561 (§2, §3.1, §3.2). Each series ends with a trailing comma, which was counted as a sample. 2560 / 256 Hz = exactly 10.0 s, so the 256 Hz conclusion is stronger.
2. **"Leave-one-format-out" in §4.3 wasn't.** `analysis/diagnostics.py` grouped by (file, format) and used `GroupKFold(3)`, which mixes arrows, letters and words in every fold. Fixed to hold out one whole format (`LeaveOneGroupOut` on format). Corrected values: Subject 3 **23.2%** (was 26.9%), Gope **22.1%** (unchanged), Supra **23.9%** (was 20.5%). The conclusion (chance) doesn't change.
3. **The §4.4 lag scan only tested three offsets.** Blocks repeat every 15 s, so lags of −5, +20 and +40 s are the same as +10, +5 and +10 s. The scan has been redone from 0 to 15 s in 0.25 s steps (`src/sanity.py`). The best balanced accuracy over 60 offsets is 51.9% / 55.4% / 56.5%, which is what picking the best of 60 noisy scores gives by chance.
4. **"Recording starts at video t = 0" can't hold everywhere.** Nine recordings are 170 s long, shorter than the 180 s video, and two are 220 s. Offsets must be set per file (`stimulus_offsets.json`).
5. **"Stimulus vs blank should be the easiest contrast" was overstated.** The blank screen is white, and the stimulus is a black symbol on white with the eyes open throughout. Much less alpha suppression is expected than for eyes open vs closed.

**Follow-up finding that changes the bottom line.** Synchronisation is not the main problem. The recordings fail basic EEG checks (`results/sanity_report.md`), in all 23 files:
- spectral slope −0.03 to −0.27 (EEG: about −1 to −2.5);
- delta/beta power ratio 1.05–1.28 (EEG: ≫ 2);
- sample-to-sample correlation 0.41–0.49 (EEG at 256 Hz: > 0.9);
- excess kurtosis ≈ 0;
- no blinks in the frontal-pole channels;
- no 50 Hz line noise in Gope's and Supra's files.

A folding test that needs no time alignment finds no response time-locked to the 15 s block cycle in any file (1 of 69 tests at p < 0.05, none after Bonferroni correction). A synthetic response of 0.3–0.5× channel SD injected into the same files is detected, so the test can find a modest effect. **The data looks like band-limited noise, so the first item in §7 P0 is now: confirm the export and the electrode contact, and record an eyes-open/eyes-closed check.**
