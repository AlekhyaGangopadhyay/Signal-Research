import mne
import numpy as np

from src.export_edf import recording_to_raw
from src.raw_io import parse_raw_txt_file


def test_edf_round_trip(raw_file, tmp_path):
    path, data = raw_file
    rec = parse_raw_txt_file(path)
    out = str(tmp_path / "x.edf")
    recording_to_raw(rec, offset_s=0.0).export(out, fmt='edf', overwrite=True, verbose='error')
    back = mne.io.read_raw_edf(out, preload=True, verbose='error')
    assert back.info['sfreq'] == 256.0
    assert back.ch_names[0] == 'FP2-F8' and len(back.ch_names) == 18
    np.testing.assert_allclose(back.get_data() * 1e6, data[:18], atol=0.05)
    assert any(d.startswith('left / arrow') for d in back.annotations.description)
