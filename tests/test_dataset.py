import numpy as np
import pytest

from src.dataset import EEGSubjectDataset, calibration_splits, shuffled_block_labels
from tests.conftest import make_subject_npz


def test_loads_stim_windows_only(subject_npz):
    ds = EEGSubjectDataset([subject_npz])
    assert len(ds) == 3 * 84
    assert set(np.unique(ds.y)) == {0, 1, 2, 3}
    assert len(ds.recordings) == 3


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_episode_without_replacement_and_disjoint_blocks(subject_npz, k):
    ds = EEGSubjectDataset([subject_npz])
    rng = np.random.default_rng(0)
    for _ in range(20):
        sx, sy, qx, qy = ds.sample_fewshot_episode(k_shot=k, q_query=5, rng=rng)
        assert sx.shape[0] == 4 * k and qx.shape[0] == 20
        # recover indices by matching rows (windows are unique random data)
        flat = ds.X.reshape(len(ds), -1)
        s_idx = [int(np.flatnonzero((flat == r.reshape(1, -1)).all(1))[0]) for r in sx.numpy()]
        q_idx = [int(np.flatnonzero((flat == r.reshape(1, -1)).all(1))[0]) for r in qx.numpy()]
        assert len(set(s_idx)) == len(s_idx) and len(set(q_idx)) == len(q_idx)
        assert not set(s_idx) & set(q_idx)
        assert not set(ds.group[s_idx]) & set(ds.group[q_idx])
        np.testing.assert_array_equal(ds.y[s_idx], sy.numpy())


def test_episode_raises_when_not_enough_blocks(tmp_path):
    ds = EEGSubjectDataset([make_subject_npz(str(tmp_path / 's.npz'), n_files=1)])
    with pytest.raises(ValueError):
        ds.sample_fewshot_episode(k_shot=20, q_query=20)


@pytest.mark.parametrize("k", [1, 5, 10])
def test_calibration_splits_are_cross_recording(subject_npz, k):
    ds = EEGSubjectDataset([subject_npz])
    splits = list(calibration_splits(ds, k, np.random.default_rng(0)))
    assert len(splits) == 3
    for s, q, r in splits:
        assert len(s) == 4 * k and len(set(s)) == len(s)
        assert set(ds.rec[s]) == {r}
        assert r not in set(ds.rec[q])
        assert np.bincount(ds.y[s], minlength=4).tolist() == [k] * 4


def test_shuffled_labels_constant_within_block(subject_npz):
    ds = EEGSubjectDataset([subject_npz])
    y = shuffled_block_labels(ds, np.random.default_rng(1))
    for g in np.unique(ds.group):
        assert len(np.unique(y[ds.group == g])) == 1
    for r in np.unique(ds.rec):
        assert sorted(y[ds.rec == r]) == sorted(ds.y[ds.rec == r])


def test_euclidean_alignment_whitens(subject_npz):
    ds = EEGSubjectDataset([subject_npz], alignment='euclidean')
    for r in np.unique(ds.rec):
        X = ds.X[ds.rec == r]
        C = np.einsum('nct,ndt->cd', X, X) / (X.shape[0] * X.shape[2])
        np.testing.assert_allclose(C, np.eye(C.shape[0]), atol=0.05)
