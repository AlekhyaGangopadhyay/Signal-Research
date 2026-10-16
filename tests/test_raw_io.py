import numpy as np
import pytest

from src.config import CHANNELS
from src.raw_io import RawFormatError, normalise_channel_name, parse_raw_txt_file, subject_from_folder
from tests.conftest import RAW_LABELS, write_raw_file


def test_parses_all_pages_in_order(raw_file):
    path, data = raw_file
    rec = parse_raw_txt_file(path)
    assert rec.n_pages == 3
    assert rec.page_samples == 2560
    assert rec.data.shape == (18, 3 * 2560)
    np.testing.assert_array_equal(rec.data, data[:18].astype(np.float32))   # aux channels dropped
    assert rec.duration_s == pytest.approx(30.0)
    assert rec.subject == 'T'
    assert rec.channel_names == CHANNELS


def test_keep_aux(raw_file):
    path, data = raw_file
    rec = parse_raw_txt_file(path, keep_aux=True)
    assert rec.data.shape[0] == 20


def test_malformed_number_raises(tmp_path):
    path = str(tmp_path / "Raw data(Subject T)" / "bad.txt")
    write_raw_file(path, corrupt='bad_number')
    with pytest.raises(RawFormatError, match="non-numeric"):
        parse_raw_txt_file(path)


def test_wrong_channel_order_raises(tmp_path):
    labels = RAW_LABELS.copy()
    labels[0], labels[1] = labels[1], labels[0]
    path = str(tmp_path / "Raw data(Subject T)" / "swapped.txt")
    write_raw_file(path, labels=labels)
    with pytest.raises(RawFormatError, match="channels"):
        parse_raw_txt_file(path)


def test_missing_channel_raises(tmp_path):
    path = str(tmp_path / "Raw data(Subject T)" / "missing.txt")
    write_raw_file(path, labels=RAW_LABELS[:-1])
    with pytest.raises(RawFormatError):
        parse_raw_txt_file(path)


def test_name_helpers():
    assert normalise_channel_name('FP2- F8') == 'FP2_F8'
    assert normalise_channel_name('F8 - T4') == 'F8_T4'
    assert subject_from_folder('Raw data(Subject Gope)') == 'Gope'
    assert subject_from_folder('Raw data(Subject 3)') == '3'
