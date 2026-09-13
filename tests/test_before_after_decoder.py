"""Opt-in single-transition decoding and compatibility with legacy checkpoints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat

ensure_hardpicks_lightning_compat()

from seismic_utils.fb_smooth import (
    decode_before_after_logits, fb_change_point_from_logits, fb_smooth_from_logits,
)
from seismic_utils.pickers import (
    attach_smooth_evaluators, before_after_decoder_from_hparams, decode_nn_picks,
    make_eval_evaluator,
)
from seismic_utils.predict import decode_fb_picks, predict_first_breaks_ms
from seismic_utils.training_state import validate_resume_checkpoint


def logits_for_boundary(boundary=200, length=400, traces=1):
    logits = torch.zeros((1, 2, traces, length))
    logits[:, 0, :, :boundary] = 5
    logits[:, 1, :, boundary:] = 5
    return logits


def hyperparams(decoder="change_point"):
    return {
        "picker": "before_after", "segm_class_count": 2,
        "segm_first_break_prob_threshold": 0.0,
        "before_after_decoder": decoder,
        "eval_metrics": [
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
            {"metric_type": "MeanAbsoluteError"},
        ],
    }


def test_rejects_isolated_false_after_samples_without_changing_legacy():
    logits = logits_for_boundary()
    logits[:, 0, :, [10, 80]] = 0
    logits[:, 1, :, [10, 80]] = 5
    original = logits.clone()
    legacy, _ = fb_smooth_from_logits(logits)
    default, _ = decode_before_after_logits(logits)
    picks, probabilities = decode_before_after_logits(logits, decoder="change_point")
    assert legacy.item() == default.item() == 10
    assert picks.item() == 200
    assert probabilities.item() == pytest.approx(torch.sigmoid(torch.tensor(5.)).item())
    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize("boundary", [0, 1, 200, 399, 400])
def test_boundaries_and_no_pick_endpoints(boundary):
    picks, probabilities = fb_change_point_from_logits(logits_for_boundary(boundary))
    expected = boundary if 0 < boundary < 400 else 0
    assert picks.item() == expected
    assert torch.isnan(probabilities).item() == (expected == 0)


def test_uniform_and_endpoint_ties_are_no_pick_and_interior_ties_are_earliest():
    uniform = torch.zeros((1, 2, 1, 20))
    assert fb_change_point_from_logits(uniform)[0].item() == 0
    # Relative costs: 0, -2, -2, 0. Both interior candidates win; choose 1.
    logits = torch.tensor([[[[0., 0., 0.]], [[-2., 0., 2.]]]])
    assert fb_change_point_from_logits(logits)[0].item() == 1
    # Relative costs: 0, 0, -2. Endpoint wins, not an observed transition.
    logits[0, 1, 0] = torch.tensor([0., -2., 0.])
    assert fb_change_point_from_logits(logits)[0].item() == 0


def test_scores_probabilities_not_only_argmax_labels():
    a = torch.tensor([[[[0., 0., 0., 0.]], [[-3., 2., -1., 3.]]]])
    b = a.clone()
    b[0, 1, 0, 1] = .5
    assert torch.equal(a.argmax(dim=1), b.argmax(dim=1))
    assert fb_change_point_from_logits(a)[0].item() == 1
    assert fb_change_point_from_logits(b)[0].item() == 3


def test_matches_brute_force_likelihood_for_uneven_lengths():
    rng = torch.Generator().manual_seed(42)
    logits = torch.randn((3, 2, 7, 40), generator=rng)
    counts = [40, 13, 1]
    picks, _ = fb_change_point_from_logits(logits, sample_counts=counts)
    for b, count in enumerate(counts):
        lp = logits[b, :, :, :count].double().log_softmax(dim=0)
        for trace in range(7):
            costs = torch.stack([
                -lp[0, trace, :k].sum() - lp[1, trace, k:].sum()
                for k in range(count + 1)
            ])
            best = costs.argmin().item()
            expected = best if 0 < best < count and costs[best] < min(costs[0], costs[-1]) else 0
            assert picks[b, trace].item() == expected


@pytest.mark.parametrize("padding", [float("nan"), float("inf"), -1000., 1000.])
def test_time_padding_never_participates(padding):
    real = logits_for_boundary(10, 20, traces=2)
    padded = torch.cat([real, torch.full((1, 2, 2, 44), padding)], dim=-1)
    actual = fb_change_point_from_logits(padded, sample_counts=[20])
    expected = fb_change_point_from_logits(real)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_real_logits_invalidate_only_affected_trace(bad):
    logits = logits_for_boundary(traces=2)
    logits[0, 0, 0, 25] = bad
    picks, probs = fb_change_point_from_logits(logits)
    assert picks.tolist() == [[0, 200]]
    assert torch.isnan(probs[0, 0]) and torch.isfinite(probs[0, 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_device_and_no_grad(dtype):
    logits = logits_for_boundary().to(dtype).requires_grad_()
    picks, probs = fb_change_point_from_logits(logits)
    assert picks.dtype == torch.long and probs.dtype == torch.float32
    assert picks.device == probs.device == logits.device
    assert picks.item() == 200 and not probs.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    logits = logits_for_boundary().half()
    for a, e in zip(fb_change_point_from_logits(logits.cuda()), fb_change_point_from_logits(logits)):
        torch.testing.assert_close(a.cpu(), e)


@pytest.mark.parametrize("counts", [[], [0], [401], [1.5], [float("nan")], [float("inf")], [20, 30]])
def test_invalid_sample_counts(counts):
    with pytest.raises(ValueError, match="sample_counts"):
        fb_change_point_from_logits(logits_for_boundary(), sample_counts=counts)


def test_invalid_inputs_and_decoder():
    with pytest.raises(TypeError):
        fb_change_point_from_logits(np.zeros((1, 2, 1, 4)))
    for shape in [(1, 2, 4), (1, 3, 1, 4), (1, 2, 1, 0)]:
        with pytest.raises(ValueError):
            fb_change_point_from_logits(torch.zeros(shape))
    with pytest.raises(TypeError):
        fb_change_point_from_logits(torch.zeros((1, 2, 1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="Unknown"):
        decode_before_after_logits(logits_for_boundary(), decoder="typo")


def test_saved_choice_override_and_legacy_default():
    hp = hyperparams()
    model = SimpleNamespace(hparams=hp)
    logits = logits_for_boundary(390)
    assert decode_nn_picks(logits, model)[0].item() == 390
    assert decode_fb_picks(logits, model, before_after_decoder="legacy")[0].item() == 0
    assert hp["before_after_decoder"] == "change_point"  # override is not a mutation
    del hp["before_after_decoder"]
    assert before_after_decoder_from_hparams(hp) == "legacy"
    assert decode_nn_picks(logits, model)[0].item() == 0
    assert decode_nn_picks(logits, model, before_after_decoder="change_point", smooth_threshold=999)[0].item() == 390
    with pytest.raises(ValueError, match="requires"):
        decode_nn_picks(logits, model, picker="fbpunet", before_after_decoder="change_point")


@pytest.mark.parametrize("decoder,expected", [("legacy", 0), ("change_point", 19)])
def test_evaluator_prediction_parity_and_padded_receiver_exclusion(decoder, expected):
    hp = hyperparams(decoder)
    logits = logits_for_boundary(19, 20, traces=2)
    logits = torch.cat([logits, torch.zeros((1, 2, 2, 12))], dim=-1)
    batch = {
        "batch_size": 1, "sample_count": torch.tensor([20]),
        "rec_ids": torch.tensor([[1, -1]]), "offset_distances": torch.zeros((1, 2, 3)),
        "origin": ["test"], "shot_id": torch.tensor([1]), "gather_id": torch.tensor([1]),
        "first_break_labels": torch.tensor([[19, -1]]),
    }
    evaluator = make_eval_evaluator(hp, "before_after")
    evaluator.ingest(batch, 0, logits)
    frame = evaluator.list_batch_dataframes[0]
    assert len(frame) == 1
    assert frame.Predictions.iloc[0] == expected
    assert frame.Errors.iloc[0] == expected - 19
    pred, _ = decode_fb_picks(logits, SimpleNamespace(hparams=hp), sample_counts=batch["sample_count"])
    assert pred[0, 0].item() == expected
    from hardpicks.metrics.base import NoneEvaluator
    model = SimpleNamespace(train_evaluator=NoneEvaluator({}))
    attach_smooth_evaluators(model, hp)
    assert model.valid_evaluator.before_after_decoder == decoder
    assert model.test_evaluator.before_after_decoder == decoder
    assert model.pred_evaluator.before_after_decoder == decoder
    assert isinstance(model.train_evaluator, NoneEvaluator)


def test_resume_old_metadata_is_legacy_but_decoder_change_is_rejected():
    saved = hyperparams("legacy")
    del saved["before_after_decoder"]
    checkpoint = {"hyper_parameters": saved, "seismic_scheduler_state": {}}
    validate_resume_checkpoint(checkpoint, hyperparams("legacy"))
    with pytest.raises(ValueError, match="before_after_decoder"):
        validate_resume_checkpoint(checkpoint, hyperparams())


def test_prediction_passes_real_sample_count_and_decoder_override():
    class Model:
        device = torch.device("cpu")
        use_dist_offsets = False
        use_first_break_prior = False
        hparams = hyperparams("legacy")

        def __call__(self, x, geom=None):
            # Collate pads 20 real samples to 32. A transition at 25 is invalid.
            return logits_for_boundary(25, x.shape[-1], x.shape[-2])

    gather = {
        "samples": np.ones((2, 20), dtype=np.float32),
        "trace_count": 2, "sample_count": 20, "sample_rate_ms": 2.,
    }
    picks = predict_first_breaks_ms(Model(), gather, before_after_decoder="change_point")
    assert np.isnan(picks).all()