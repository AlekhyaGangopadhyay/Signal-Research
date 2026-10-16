import numpy as np
import pytest
import torch

from src.dataset import EEGSubjectDataset
from src.inference import DirectionPredictor
from src.losses import compute_prototypes
from src.models import EEGNetBackbone, SpatialTemporalProtoNet, adapt_batchnorm, build_model
from src.train_fewshot import train_meta_model
from tests.conftest import make_subject_npz


@pytest.mark.parametrize("name", ["spatial_temporal", "eegnet"])
def test_model_output_is_unit_norm(name):
    m = build_model(name).eval()
    z = m(torch.randn(5, 18, 512))
    assert z.shape == (5, 64)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(5))


def test_spatial_conv_is_depthwise():
    m = SpatialTemporalProtoNet()
    assert m.conv_spatial.groups == m.conv_spatial.in_channels


def test_adabn_changes_running_stats():
    m = EEGNetBackbone()
    before = m.bn1.running_mean.clone()
    adapt_batchnorm(m, np.random.default_rng(0).standard_normal((64, 18, 512)).astype(np.float32) * 5 + 3)
    assert not torch.allclose(before, m.bn1.running_mean)
    assert not m.training


def test_prototypes_use_labels():
    z = torch.eye(4)
    y = torch.tensor([3, 1, 0, 2])
    protos = compute_prototypes(z, y, 4)
    torch.testing.assert_close(protos[3], z[0])


def test_calibrate_is_order_independent(tmp_path, subject_npz):
    ds = EEGSubjectDataset([subject_npz])
    ckpt = str(tmp_path / "m.pth")
    train_meta_model(ds, epochs=1, episodes_per_epoch=2, save_path=ckpt, verbose=False)
    idx = np.concatenate([np.flatnonzero(ds.y == c)[:3] for c in range(4)])
    a, b = DirectionPredictor(ckpt), DirectionPredictor(ckpt)
    a.calibrate(ds.X[idx], ds.y[idx])
    perm = np.random.default_rng(0).permutation(len(idx))
    b.calibrate(ds.X[idx][perm], [['left', 'right', 'forward', 'backward'][i] for i in ds.y[idx][perm]])
    torch.testing.assert_close(a.prototypes, b.prototypes)
    with pytest.raises(ValueError):
        a.calibrate(ds.X[idx[:3]], ds.y[idx[:3]])          # only one direction


def test_training_learns_a_real_signal(tmp_path):
    """With a strong class-dependent signal the pipeline must beat chance (guards against silent bugs)."""
    from src.evaluate import evaluate_fewshot_on_target
    train = EEGSubjectDataset([make_subject_npz(str(tmp_path / 'a.npz'), seed=1, signal=1.0),
                               make_subject_npz(str(tmp_path / 'b.npz'), seed=2, signal=1.0)])
    target = EEGSubjectDataset([make_subject_npz(str(tmp_path / 'c.npz'), seed=3, signal=1.0)])
    model, _ = train_meta_model(train, epochs=6, episodes_per_epoch=20, verbose=False)
    res = evaluate_fewshot_on_target(model, target, 5, np.random.default_rng(0))
    assert res['accuracy'] > 0.5 and res['p_value'] < 0.01
