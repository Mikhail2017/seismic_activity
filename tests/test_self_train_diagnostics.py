from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from seismic_utils.minimal_preprocess import MinimalAnnotationDataset, minimal_batch_collate
from seismic_utils.minimal_split import (
    assert_unlabeled_item_has_no_gt, gather_key, load_split, make_minimal_split,
    split_keys, write_split,
)
from seismic_utils.self_train_diagnostics import (
    AnnotationSourceDataset, SelfTrainDiagnostics, batch_pixel_counts, content_digest,
    describe_geometry, effective_protocol, load_pseudo_archive, merge_counts,
    prediction_counts, split_identity, write_json, write_pseudo_shard,
)


class FakeParser:
    def __init__(self, size=10):
        self.items = []
        for i in range(size):
            labels = np.array([6, 7, 0, 9], dtype=np.int32)
            self.items.append({
                "origin": "Sudbury", "gather_id": i, "shot_id": i + 100, "rec_line_id": 1,
                "trace_count": 4, "sample_count": 24, "sample_rate_ms": 2.0,
                "samples": np.random.default_rng(i).normal(size=(4, 24)).astype(np.float32),
                "first_break_labels": labels, "first_break_timestamps": labels * 2.0,
                "bad_first_breaks_mask": labels <= 0, "rec_ids": np.arange(4) + 1,
                "offset_distances": np.tile(np.arange(4, dtype=np.float32)[:, None], (1, 3)),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return copy.deepcopy(self.items[index])

    def get_meta_gather(self, index):
        return {k: copy.deepcopy(v) for k, v in self.items[index].items() if k != "samples"}


def test_strict_json_and_split_identity(tmp_path):
    keys = [gather_key(item) for item in FakeParser().items]
    split = make_minimal_split(keys, site="Sudbury", seed=4, n_labeled=4)
    write_split(split, tmp_path / "split.json")
    loaded = load_split(tmp_path / "split.json")
    assert split_identity(split) == split_identity(loaded)
    for name in ("labeled_train", "labeled_val", "unlabeled_pool"):
        assert split_keys(split, name) == split_keys(loaded, name)
    assert content_digest({"a": 1, "b": 2}) == content_digest({"b": 2, "a": 1})
    write_json(tmp_path / "strict.json", {"missing": np.nan, "x": torch.tensor([1., float("inf")])})
    text = (tmp_path / "strict.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"missing": None, "x": [1., None]}
    assert not list(tmp_path.glob("*.tmp"))


def test_actual_protocol_and_metadata_geometry():
    cfg = {"loss_type": "crossentropy", "loss_params": {"weight": [1, 100]},
           "minimal_annotations": {"pad_multiple": 128, "label_window_ms": 3}}
    protocol = effective_protocol(cfg, 10., "valid/HitRate1px")
    assert protocol["window"]["half_width"] == 10
    assert protocol["padding"]["multiple"] == 16
    assert protocol["padding"]["fixed_shape"] is None
    assert protocol["loss"]["normalization"] == "sum_target_class_weights"
    assert protocol["checkpoint"]["continuation"] == "best_weights_only"
    assert protocol["decoder"]["masks_padding"] is False
    parser = FakeParser(2)
    parser.items[1]["sample_rate_ms"] = 1.0
    # Geometry must not read amplitudes or use annotation content.
    parser.items[1]["first_break_labels"] = None
    geometry = describe_geometry(parser, [0, 1], 10.)
    assert geometry["batch_padding_upper_bound"] == [16, 32]
    assert {row["half_width_samples"] for row in geometry["training_windows"]} == {5, 10}


def test_pixel_counts_use_actual_targets_and_preserve_oracle():
    parser = FakeParser(2)
    key = gather_key(parser.items[1])
    manual = MinimalAnnotationDataset(parser, [0], mode="labeled")
    pseudo = MinimalAnnotationDataset(parser, [1], mode="pseudo", pseudo_picks={
        key: np.array([12., np.nan, 13., 14.]),
    })
    items = [AnnotationSourceDataset(manual, 0)[0], AnnotationSourceDataset(pseudo, 1)[0]]
    batch = minimal_batch_collate(items)
    counts = batch_pixel_counts(batch)
    for name, item in zip(("manual", "pseudo"), items):
        mask = item["segmentation_mask"]
        assert counts[name]["positive"] == int((mask == 1).sum())
        assert counts[name]["background"] == int((mask == 0).sum())
        assert counts[name]["ignored_real"] == int((mask == -1).sum())
        assert counts[name]["ignored_padding"] == 16 * 32 - 4 * 24
        assert counts[name]["ignored"] == counts[name]["ignored_real"] + counts[name]["ignored_padding"]
    assert merge_counts([counts, counts])["pseudo"]["positive"] == 2 * counts["pseudo"]["positive"]
    oracle = MinimalAnnotationDataset(parser, [1], mode="oracle")[0]
    np.testing.assert_array_equal(oracle["first_break_labels"], parser.items[1]["first_break_labels"])
    assert not np.array_equal(oracle["first_break_labels"], items[1]["first_break_labels"])


def test_prediction_diagnostics_do_not_clean_picks():
    picks = np.array([0., -1., np.nan, np.inf, 3., 23., 24., 30.])
    before = picks.copy()
    counts = prediction_counts(picks, 24)
    assert counts == {"n_traces": 8, "n_predicted": 4, "n_nonfinite": 2,
                      "n_unpicked_or_nonpositive": 2, "n_out_of_range": 2, "n_invalid": 6}
    np.testing.assert_array_equal(picks, before)


def test_pseudo_archive_roundtrip_and_integrity(tmp_path):
    key = ("Sudbury", 1, 2, 3)
    picks = np.array([9., np.nan, 31.])  # preserve out-of-range raw QC results in stage 0
    qc = [{"key": list(key), "admit": True, "receiver_ids": [10, 11, 12]}]
    shard = write_pseudo_shard(tmp_path / "shard.json", {key: picks}, qc,
                              iteration=3, teacher=tmp_path / "best.ckpt", split_sha256="split")
    index_path = write_json(tmp_path / "index.json", {"split_sha256": "split", "shards": [shard]})
    restored, provenance = load_pseudo_archive(index_path)
    np.testing.assert_array_equal(restored[key], picks)
    assert provenance[key]["iteration"] == 3
    assert provenance[key]["teacher_checkpoint"] == str(tmp_path / "best.ckpt")
    assert provenance[key]["qc"]["receiver_ids"] == [10, 11, 12]
    with pytest.raises(ValueError, match="already exists"):
        write_pseudo_shard(tmp_path / "shard.json", {}, [], iteration=4,
                           teacher=tmp_path / "best.ckpt", split_sha256="split")
    (tmp_path / "shard.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_pseudo_archive(index_path)


def test_qc_inference_strips_gt_and_records_padding_bug():
    from train import fbp_self_train as cli

    class PaddingModel(torch.nn.Module):
        def _prepare_input_features(self, batch):
            assert torch.all(batch["first_break_labels"] == 0)
            assert torch.all(batch["first_break_timestamps"] == 0)
            assert torch.all(batch["segmentation_mask"] == -1)
            return batch["samples"]

        def forward(self, samples):
            logits = torch.zeros(samples.shape[0], 2, *samples.shape[1:])
            logits[:, 1, :, -1] = 10  # sample 31 is padding; do not fix in stage 0
            return logits

    parser = FakeParser(1)
    dataset = MinimalAnnotationDataset(parser, [0], mode="unlabeled")
    assert_unlabeled_item_has_no_gt(dataset[0])
    accepted, stats = cli._infer_qc(PaddingModel(), dataset, torch.device("cpu"))
    assert stats[0]["n_out_of_range"] == 4
    assert stats[0]["n_admitted_out_of_range"] == 4
    assert stats[0]["n_survive"] == 4
    np.testing.assert_array_equal(accepted[gather_key(parser.items[0])], [31] * 4)
    np.testing.assert_array_equal(parser.items[0]["first_break_labels"], [6, 7, 0, 9])


def test_validation_diagnostics_ignore_sanity_and_match_report(tmp_path):
    from types import SimpleNamespace
    from seismic_utils.fbp_eval_report import paper_pick_metrics

    frame = pd.DataFrame({"Errors": [0., 10., -5., np.nan], "Predictions": [3, 20, 0, 8]})
    evaluator = SimpleNamespace(_dataframe=frame, finalize=lambda: None)
    model = SimpleNamespace(valid_evaluator=evaluator)
    trainer = SimpleNamespace(sanity_checking=True, current_epoch=0)
    callback = SelfTrainDiagnostics(tmp_path)
    callback.on_validation_end(trainer, model)
    assert not callback.epochs
    trainer.sanity_checking = False
    callback.on_validation_end(trainer, model)
    metrics = callback.epochs[0]["validation"]
    for key, value in paper_pick_metrics(frame).items():
        assert metrics[key] == value
    assert metrics["MAE"] == 5
    assert metrics["W_total_10"] == pytest.approx(2 / 3)
    assert metrics["W_total_10"] == metrics["Coverage"] * metrics["W_pred_10"]


def test_main_cpu_training_artifacts_and_reused_split(tmp_path, monkeypatch):
    """Real Lightning fits/evaluation, tiny U-Net; fixed QC output forces growth/reset."""
    from train import fbp_self_train as cli

    parser = FakeParser()
    original = copy.deepcopy(parser.items)
    split = make_minimal_split([gather_key(i) for i in parser.items], site="Sudbury", seed=7, n_labeled=4)
    split_path = write_split(split, tmp_path / "saved_split.json")
    recipe = cli._load_recipe(cli.DEFAULT_CONFIG)
    recipe["minimal_annotations"]["label_window_ms"] = 3  # overridden by current entry point
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(yaml.safe_dump(recipe))
    build_config = cli.train_cli.build_model_config

    def tiny_config(**kwargs):
        cfg, label = build_config(**kwargs)
        cfg.update(encoder_block_channels=[4, 8, 16, 32], mid_block_channels=32,
                   decoder_block_channels="[32, 16, 8, 4]")
        return cfg, label

    def controlled_qc(model, dataset, device):
        accepted, stats = {}, []
        for i in range(len(dataset)):
            item = dataset[i]
            assert_unlabeled_item_has_no_gt(item)
            key = gather_key(item)
            pred = np.array([12., np.nan, 13., 14.])
            accepted[key] = pred
            stats.append({"key": list(key), "admit": True, "survive_frac": .75,
                          **prediction_counts(pred, 24), "n_survive": 3,
                          "n_admitted_out_of_range": 0, "receiver_ids": item["rec_ids"].tolist()})
        return accepted, stats

    captured = {}

    def report(**kwargs):
        captured.update(kwargs)
        return kwargs["report_dir"]

    monkeypatch.setattr(cli.train_cli, "build_model_config", tiny_config)
    monkeypatch.setattr(cli.train_cli, "build_split_parser", lambda *a, **kw: parser)
    monkeypatch.setattr(cli, "_infer_qc", controlled_qc)
    monkeypatch.setattr(cli, "_write_99pct_report", report)
    monkeypatch.setattr(cli, "RESET_AFTER", {1})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SEISMIC_RUN_ID", "stage0_test")
    out = tmp_path / "run"
    args = ["--config", str(config_path), "--sites", "Sudbury", "--ablation", "iterative",
            "--seed", "0", "--split-json", str(split_path), "--output-dir", str(out),
            "--n-iters", "3", "--epochs", "2", "--n-draw", "2", "--batch-size", "2",
            "--num-workers", "0", "--accelerator", "cpu", "--devices", "1"]
    assert cli.main(args) == 0
    effective = json.loads((out / "effective_config.json").read_text())
    assert yaml.safe_load((out / "effective_config.yaml").read_text()) == effective
    assert effective["seed"] == 0 and effective["split"]["seed"] == 7
    assert effective["split"]["sha256"] == content_digest(split)
    assert load_split(out / "split.json") == split
    assert effective["requested_recipe"]["minimal_annotations"]["label_window_ms"] == 3
    assert effective["protocol"]["window"]["half_width"] == 10
    assert effective["model_config"]["minimal_annotations"]["label_window_ms"] == 10
    assert effective["geometry"]["labeled_train"]["batch_padding_upper_bound"] == [16, 32]
    assert effective["code"]["revision"]
    history = json.loads((out / "iteration_history.json").read_text())
    assert [r["cycle"] for r in history] == [1, 2, 2]
    assert [r["n_pseudo_train"] for r in history] == [0, 2, 4]
    assert [r["n_admitted"] for r in history] == [2, 2, 0]
    assert history[0]["weight_reset"]
    for row in history:
        it = row["iteration"]
        fit = json.loads((out / f"iter_{it:02d}" / "fit_diagnostics.json").read_text())
        epochs = json.loads((out / f"iter_{it:02d}" / "epoch_diagnostics.json").read_text())
        assert len(fit["epochs"]) == 2  # sanity validation not included
        assert epochs["epochs"] == fit["epochs"]
        assert fit["runtime"]["loss"]["weight"] == [1., 100.]
        assert fit["runtime"]["loss"]["reduction"] == "mean"
        assert fit["pixels"]["manual"]["gather_exposures"] == 6
        assert fit["pixels"]["pseudo"]["gather_exposures"] == 2 * row["n_pseudo_train"]
        assert fit["terminal_optimizers"][0]["step_max"] == fit["terminal_global_step"]
        assert row["validation"]["n_labeled"] == 3
        assert row["validation"] == fit["epochs"][fit["selected_epoch"]]["validation"]
        ckpt_config = yaml.safe_load((out / f"iter_{it:02d}" / "model_config.yaml").read_text())
        assert ckpt_config == effective["model_config"]
    picks, provenance = load_pseudo_archive(out / "pseudo_labels.json")
    assert len(picks) == 4
    assert set(picks) <= set(split_keys(split, "unlabeled_pool"))
    assert {p["iteration"] for p in provenance.values()} == {1, 2}
    # The final evaluator sees manual picks even for the four admitted gathers.
    traces = captured["traces"]
    assert len(traces) == 6 * 4
    for gid, frame in traces.groupby("GatherId"):
        expected = parser.items[int(gid)]["first_break_labels"]
        for _, row in frame.iterrows():
            target = expected[int(row["ReceiverId"]) - 1]
            if target > 0:
                assert row["Predictions"] - row["Errors"] == target
            else:
                assert pd.isna(row["Errors"])
    for item, old in zip(parser.items, original):
        np.testing.assert_array_equal(item["first_break_labels"], old["first_break_labels"])
        np.testing.assert_array_equal(item["samples"], old["samples"])
    # A second invocation must not overwrite even the top-level split/config.
    before = (out / "split.json").read_bytes()
    with pytest.raises(ValueError, match="not empty"):
        cli.main(args)
    assert (out / "split.json").read_bytes() == before


def test_instrumentation_does_not_change_training(tmp_path, monkeypatch):
    from train import fbp_self_train as cli

    cli.ensure_hardpicks_lightning_compat()
    monkeypatch.setenv("SEISMIC_RUN_ID", "stage0_parity")
    parser = FakeParser(4)
    train = AnnotationSourceDataset(MinimalAnnotationDataset(parser, [0, 1, 2], mode="labeled"), 0)
    valid = MinimalAnnotationDataset(parser, [3], mode="oracle")
    cfg, _ = cli.train_cli.build_model_config(
        model="meneses", max_epochs=2, recipe=cli._load_recipe(cli.DEFAULT_CONFIG), picker="fbpunet",
    )
    cfg = cli._apply_ablation(cfg, "combined", 10.)
    cfg.update(encoder_block_channels=[4, 8, 16, 32], mid_block_channels=32,
               decoder_block_channels="[32, 16, 8, 4]")
    params = dict(model_config=cfg, train_ds=train, valid_ds=valid, epochs=2, batch_size=2,
                  num_workers=1, init_ckpt=None, devices=1, accelerator="cpu", precision="32", seed=123)
    observed = cli._inner_fit(output_dir=tmp_path / "observed", **params)
    make_trainer = cli.train_cli.make_trainer

    def without_diagnostics(**kwargs):
        kwargs["callbacks"] = [cb for cb in kwargs["callbacks"] if not isinstance(cb, SelfTrainDiagnostics)]
        return make_trainer(**kwargs)

    monkeypatch.setattr(cli.train_cli, "make_trainer", without_diagnostics)
    baseline = cli._inner_fit(output_dir=tmp_path / "baseline", **params)
    a, b = cli.load_checkpoint(observed), cli.load_checkpoint(baseline)
    assert a["epoch"] == b["epoch"] and a["global_step"] == b["global_step"]
    for key in a["state_dict"]:
        torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    for key, values in a["optimizer_states"][0]["state"].items():
        for field, value in values.items():
            torch.testing.assert_close(value, b["optimizer_states"][0]["state"][key][field], rtol=0, atol=0)
    assert a["seismic_validation_metrics"] == b["seismic_validation_metrics"]