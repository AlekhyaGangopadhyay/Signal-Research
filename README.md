# Few-Shot EEG Direction Detection

Pipeline for decoding 4 viewed directions (`left`, `right`, `forward`, `backward`) from 18-channel bipolar EEG,
with few-shot calibration on new subjects (prototypical network), plus classical baselines and data-validity checks.

## Status (2026-10-10)

**The current recordings do not contain detectable EEG.** All 23 files have a flat spectrum, Gaussian amplitudes,
no eye blinks and no response time-locked to the stimulus (`results/sanity_report.md`). Every baseline and the deep
model are at chance, and the shuffled-label controls score the same. Earlier accuracies of up to 55% came from
labelling files by their order and from test trials that duplicated calibration trials. See
[CRITICAL_REVIEW.md](CRITICAL_REVIEW.md) for the analysis and [WORKLOG.md](WORKLOG.md) for every change made since.

The code is ready for valid data. What's needed on the recording side is in
[docs/DATA_CONVERSION_AND_EXECUTION_GUIDE.md §8](docs/DATA_CONVERSION_AND_EXECUTION_GUIDE.md#8-what-the-data-needs-before-decoding-can-work).

## Project structure

```
Signal-Research/
├── Raw data/                  raw exports, one folder per subject, one .txt per recording
├── Stimuli/                   stimulus video (12 blocks: arrows, letters, words)
├── stimulus_offsets.json      per-file video start inside each recording (currently unverified 0.0)
├── src/
│   ├── config.py              paths, 256 Hz signal settings, training defaults, set_seed()
│   ├── raw_io.py              strict parser for the raw exports (all pages, channel checks)
│   ├── stimulus.py            video timeline, per-file offsets, stimulus-locked windows
│   ├── preprocessing.py       filters (offline + causal), robust scaling, Euclidean alignment, artifact flags
│   ├── convert_raw_data.py    raw -> processed_data/tensors/subject_*.npz + metadata.csv
│   ├── sanity.py              is it EEG? alignment-free folding test, lag scan, positive control
│   ├── dataset.py             episodes without replacement, cross-recording calibration splits
│   ├── baselines.py           band power, FBCSP, Riemannian tangent space / MDM
│   ├── stats.py               block-bootstrap CIs, block-permutation tests
│   ├── models.py              multi-scale EEGNet-style ProtoNet (7k params), EEGNet, AdaBN
│   ├── losses.py              prototypical loss, optional directional angular loss
│   ├── train_fewshot.py       episodic training with early stopping on held-out recordings
│   ├── evaluate.py            LOSO few-shot evaluation with shuffled-label control
│   └── inference.py           label-aware calibration, checkpoint-driven config, streaming predictor
├── tests/                     pytest suite (37 tests, synthetic data)
├── analysis/                  scripts behind CRITICAL_REVIEW.md
├── docs/                      data guide and research plan
├── run_pipeline.py            runs sanity -> conversion -> baselines -> LOSO
└── WORKLOG.md                 timestamped log of changes
```

Generated folders (`processed_data/`, `checkpoints/`, `results/`) are not tracked by git.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Python >= 3.10
.venv/Scripts/python -m pytest tests              # 37 tests
```

## Running

```bash
python -m src.sanity                      # 1. check the data is EEG and stimulus-related (do this first)
python -m src.convert_raw_data            # 2. build stimulus-locked epochs (add --csv for per-window CSVs)
python -m src.baselines                   # 3. classical baselines with CIs, p-values and shuffled controls
python -m src.evaluate --shots 1 3 5 10   # 4. LOSO few-shot ProtoNet (+ shuffled-label control model)
python run_pipeline.py                    #    or all four in order
```

Useful options: `src.evaluate --model eegnet`, `--alignment none`, `--no-adabn`, `--angular` (for the ablation).

## Inference

```python
import numpy as np
from src.inference import DirectionPredictor, StreamingPredictor

predictor = DirectionPredictor("checkpoints/protonet_spatial_temporal.pth")   # from python -m src.train_fewshot

# Calibrate with preprocessed windows [N, 18, 512] from the new subject, labels in any order:
predictor.calibrate(support_windows, ["left", "right", "forward", "backward", ...])
direction, confidences = predictor.predict_trial(window)                      # window: [18, 512]

# Real time: causal filtering of a raw stream
stream = StreamingPredictor(predictor, calibration_raw=raw_recording)         # raw [18, T]
stream.push(raw_chunk)                                                        # [18, n] as it arrives
if stream.ready():
    print(stream.predict())
```

## Evaluation protocol

- Labels come from the stimulus timeline. Each recording contains all 4 directions, 3 times each.
- No sampling with replacement. Support and query come from different stimulus blocks.
- Few-shot: calibrate on one recording of the unseen subject, test on all of its other recordings.
- Every number is reported with a block-bootstrap 95% CI and a block-permutation p-value, next to a shuffled-label control and classical baselines.
