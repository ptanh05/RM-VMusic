import copy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import load_config, MODALITIES
from src.pipeline import build_model
from src.models.uad_fusion import UADFusionModel
from src.data.dataset import FeatureTensorDataset
from src.training.trainer import batch_inputs, predict_model, set_seed, fit_model
from src.training.losses import SupervisedContrastiveLoss


def inputs():
    # Model input tensors are test fixtures, not physical feature caches.
    torch.manual_seed(3)
    return [torch.randn(4, d) for d in (5000, 512, 128)] + [
        torch.tensor([1., 1., 0., 0.]), torch.tensor([1., 0., 1., 0.]), torch.tensor([1., 0., 0., 0.])]


def test_uad_forward():
    model = UADFusionModel(proj_dim=8, dropout=0).eval()
    out = model(*inputs())
    assert out["logits"].shape == (4, 12)
    assert torch.isfinite(out["logits"]).all()
    assert out["has_evidence"].tolist() == [True, True, True, False]
    assert out["fused_embedding"][3].count_nonzero() == 0


def test_uad_reliability_range_and_formula():
    model = UADFusionModel(proj_dim=8, dropout=0).eval()
    out = model(*inputs())
    u = torch.stack([out["uncertainty"][m] for m in MODALITIES], dim=1)
    r = torch.stack([out["reliability"][m] for m in MODALITIES], dim=1)
    assert (u >= 0).all() and ((r >= 0) & (r <= 1)).all()
    torch.testing.assert_close(r, torch.exp(-u) * out["active_masks"])
    torch.testing.assert_close(out["modality_weights"][:3], (r / r.sum(1, keepdim=True).clamp_min(1e-8))[:3])
    assert (out["modality_weights"][out["active_masks"] == 0] == 0).all()
    torch.testing.assert_close(out["modality_weights"].sum(1), torch.tensor([1., 1., 1., 0.]))
    expected = sum(out["fusion_weights"][m][:, None] * out["embeddings"][m] for m in MODALITIES)
    torch.testing.assert_close(expected, out["fused_embedding"])


def test_uad_extreme_uncertainty_and_gradient():
    model = UADFusionModel(proj_dim=8, dropout=0).eval()
    for mod in MODALITIES:
        with torch.no_grad():
            getattr(model, f"{mod}_rel").net[-1].bias.fill_(10000)
    out = model(*inputs())
    assert torch.isfinite(out["logits"]).all()
    torch.testing.assert_close(out["modality_weights"][:3].sum(1), torch.ones(3))
    out["logits"].square().sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_uncertainty_receives_gradients():
    model = UADFusionModel(proj_dim=8, dropout=0).eval()
    model(*inputs())["logits"][0].square().sum().backward()
    for mod in MODALITIES:
        grads = [p.grad for p in getattr(model, f"{mod}_rel").parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_modality_dropout():
    mask = torch.tensor([[1., 1., 1.], [1., 0., 0.], [0., 0., 0.]])
    model = UADFusionModel(proj_dim=8, modality_dropout_p=1).train()
    torch.testing.assert_close(model.apply_modality_dropout(mask), mask)
    model.modality_dropout_p = .5
    many = mask.repeat(100, 1)
    torch.manual_seed(42)
    dropped = model.apply_modality_dropout(many)
    assert (dropped <= many).all()
    assert ((dropped.sum(1) > 0) == (many.sum(1) > 0)).all()
    assert not torch.equal(dropped, many)
    model.eval()
    torch.testing.assert_close(model.apply_modality_dropout(mask), mask)


def test_missing_inputs_do_not_affect_predictions():
    model = UADFusionModel(proj_dim=8, dropout=0).eval()
    a = inputs()
    first = model(*a)["logits"].detach()
    for i in range(3):
        a[i][a[i+3] == 0] = 1000
    torch.testing.assert_close(model(*a)["logits"], first)


def test_old_checkpoint_fails_explicitly():
    model = UADFusionModel(proj_dim=8)
    state = model.state_dict()
    del state["architecture_version"]
    with pytest.raises(ValueError, match="checkpoint"):
        model.load_state_dict(state, strict=False)


@pytest.mark.parametrize("name", ["lyrics_only", "cover_only", "audio_only", "early_concat", "late_fusion", "uad"])
def test_all_model_apis(name, bundle, small_metadata):
    config = load_config()
    config["model"]["proj_dim"] = 8
    ds = FeatureTensorDataset.from_bundle(small_metadata, bundle)
    out = predict_model(build_model(name, config), DataLoader(ds, batch_size=2))
    assert out["probabilities"].shape == (4, 12)
    assert out["song_ids"] == small_metadata.song_id.tolist()
    np.testing.assert_allclose(out["probabilities"].sum(1), 1, atol=1e-6)


def test_stress_missingness_and_repeatable_inference(bundle, small_metadata):
    ds = FeatureTensorDataset.from_bundle(small_metadata, bundle)
    model = UADFusionModel(proj_dim=8).eval()
    a = predict_model(model, DataLoader(ds, batch_size=1), missing_rate=.5, mask_seed=9)
    b = predict_model(model, DataLoader(ds, batch_size=4), missing_rate=.5, mask_seed=9)
    np.testing.assert_array_equal(a["masks"], b["masks"])
    np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-6)
    all_missing = predict_model(model, DataLoader(ds), missing_rate=1)
    assert not all_missing["masks"].any() and not all_missing["weights"].any()


def test_unpaired_invariance_rejected_before_training():
    with pytest.raises(ValueError, match="invariance"):
        fit_model(None, None, None, lambda_inv=.05)


def test_supcon_without_positive_pairs():
    x = torch.ones(3, 8, requires_grad=True)
    loss = SupervisedContrastiveLoss()(x, torch.tensor([0, 1, 2]))
    assert loss == 0
    loss.backward()
    assert torch.isfinite(x.grad).all()
