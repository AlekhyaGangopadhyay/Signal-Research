import numpy as np
from scipy.signal import welch

from src.preprocessing import CausalPreprocessor, preprocess_continuous_eeg, flag_artifacts
from src.sanity import folding_test, signal_quality
from src.stats import block_permutation_test, cluster_bootstrap_ci

FS = 256.0


def test_notch_removes_50hz_at_256():
    t = np.arange(int(20 * FS)) / FS
    x = np.vstack([np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 10 * t)] * 2)
    y = preprocess_continuous_eeg(x)
    f, p = welch(y[0], fs=FS, nperseg=1024)
    assert p[np.argmin(abs(f - 50))] < 1e-3 * p[np.argmin(abs(f - 10))]


def test_causal_preprocessor_ignores_future():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((18, 4096))
    a = CausalPreprocessor(18).fit(raw)
    b = CausalPreprocessor(18).fit(raw)
    x1, x2 = raw.copy(), raw.copy()
    x2[:, 3000:] += 100.0                       # change only the future
    out1 = np.concatenate([a.process(x1[:, i:i + 128]) for i in range(0, 4096, 128)], axis=1)
    out2 = np.concatenate([b.process(x2[:, i:i + 128]) for i in range(0, 4096, 128)], axis=1)
    np.testing.assert_allclose(out1[:, :3000], out2[:, :3000])


def test_flag_artifacts():
    X = np.random.default_rng(0).standard_normal((3, 18, 512)).astype(np.float32)
    X[1, 0, 100] = 50.0
    X[2, 5] = 0.0
    bad, reasons = flag_artifacts(X)
    assert bad.tolist() == [False, True, True]
    assert reasons[1] == 'amplitude' and reasons[2] == 'flat'


def test_signal_quality_flags_white_noise_and_accepts_pink():
    rng = np.random.default_rng(0)
    white = rng.standard_normal((18, int(60 * FS)))
    assert signal_quality(white)['verdict'] == 'noise-like'
    # 1/f^1.5 noise with occasional large frontal blinks, roughly EEG-like
    n = int(60 * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    spec = (rng.standard_normal((18, len(f))) + 1j * rng.standard_normal((18, len(f)))) / np.maximum(f, 0.5) ** 0.75
    pink = np.fft.irfft(spec, n=n)
    pink /= pink.std(axis=1, keepdims=True)
    for start in rng.integers(0, n - 100, 30):
        pink[[0, 4, 8, 12], start:start + 100] += 8 * np.hanning(100)
    assert signal_quality(pink)['verdict'] == 'EEG-like'


def test_folding_detects_periodic_response_not_noise():
    rng = np.random.default_rng(0)
    n = int(180 * FS)
    noise = rng.standard_normal(n)
    assert folding_test(noise, n_perm=200, rng=rng)[1] > 0.01
    t = np.arange(n) / FS
    periodic = noise + 1.0 * ((t % 15) < 1.0)
    assert folding_test(periodic, n_perm=200, rng=rng)[1] < 0.01


def test_permutation_test_and_ci():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(48), 7)
    labels = np.repeat(np.tile(np.arange(4), 12), 7)
    acc, p, null = block_permutation_test(labels.copy(), labels, groups, n_perm=500, rng=rng)
    assert acc == 1.0 and p < 0.01 and abs(null - 0.25) < 0.05
    random_preds = rng.integers(0, 4, len(labels))
    _, p_rand, _ = block_permutation_test(random_preds, labels, groups, n_perm=500, rng=rng)
    assert p_rand > 0.01
    lo, hi = cluster_bootstrap_ci(random_preds == labels, groups, rng=rng)
    assert lo < 0.25 < hi or abs((lo + hi) / 2 - 0.25) < 0.08
