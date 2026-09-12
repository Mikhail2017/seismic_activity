"""Small offline fixtures shared by training consistency tests."""

import h5py
import mlflow
import numpy as np
import pytest

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat


@pytest.fixture(autouse=True)
def offline_model_logging(monkeypatch):
    ensure_hardpicks_lightning_compat()
    monkeypatch.setattr(mlflow, "log_param", lambda *args, **kwargs: None)


@pytest.fixture
def tiny_hdf5(tmp_path):
    path = tmp_path / "tiny.hdf5"
    count, traces, samples = 8, 4, 128
    shots = np.repeat(np.arange(1, count + 1), traces)
    fields = {
        "REC_PEG": np.tile(np.arange(1001, 1001 + traces), count),
        "REC_X": np.tile(np.arange(traces) * 50 + 3000, count),
        "REC_Y": np.zeros(count * traces), "REC_HT": np.zeros(count * traces),
        "SAMP_NUM": np.full(count * traces, samples), "SAMP_RATE": np.full(count * traces, 2000),
        "COORD_SCALE": np.ones(count * traces), "HT_SCALE": np.ones(count * traces),
        "SHOTID": shots, "SHOT_PEG": shots,
        "SOURCE_X": np.zeros(count * traces), "SOURCE_Y": np.zeros(count * traces),
        "SOURCE_HT": np.zeros(count * traces), "SPARE1": np.tile([40., 80., 120., 160.], count),
        "data_array": np.random.default_rng(0).normal(size=(count * traces, samples)).astype(np.float32),
    }
    with h5py.File(path, "w") as f:
        group = f.create_group("TRACE_DATA/DEFAULT")
        for key, value in fields.items():
            group.create_dataset(key, data=value if key == "data_array" else value[:, None])
    return dict(processed_hdf5_path=str(path), site_name="synthetic",
                receiver_id_digit_count=3, first_break_field_name="SPARE1")


def make_parser(info, **site_params):
    from seismic_utils.hdf5_parser import create_hdf5_parser

    return create_hdf5_parser(
        site_info=info, site_params=site_params, prefix="train" if site_params.get("augmentations") else "valid",
        dataset_hyper_params=dict(convert_to_fp16=True, convert_to_int16=True,
                                 preload_trace_data=False, cache_trace_metadata=True, provide_offset_dists=True),
        segm_class_count=2,
    )


def make_model(max_epochs=3):
    from models.fbp.unet import FBPUNet
    from seismic_utils.pickers import attach_smooth_evaluators
    from train.fbp_train import build_model_config

    hp, _ = build_model_config(model="vanilla-before-after", max_epochs=max_epochs, smooth_threshold=7)
    hp.update(encoder_block_count=2, encoder_block_channels=[4, 8],
              mid_block_channels=8, decoder_block_channels=[8, 4])
    model = FBPUNet(hp)
    attach_smooth_evaluators(model, hp)
    return model