"""
Smoke test for Checkpoint Ensemble and TTA uncertainty — no real data needed.
Runs entirely on random tensors that match the expected input shape.

Usage:
    python test_uncertainty.py
"""

import torch
import sys

# ---- Config matching your training setup ----
BATCH      = 2
T          = 4      # use a short sequence instead of 18 for speed
H, W       = 224, 224
NUM_CLASS  = 8
N_MODELS   = 3     # number of checkpoints in the ensemble
N_TTA      = 3     # use 3 TTA transforms

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


def make_dummy_input():
    """Random (B, 1, H, W) sequence of length T."""
    return [torch.randn(BATCH, 1, H, W, device=DEVICE) for _ in range(T)]


def make_dummy_models(n):
    """Instantiate n randomly-initialised models (simulates different checkpoints)."""
    from models.reslstmunet import ResLSTMUNet
    models = []
    for _ in range(n):
        m = ResLSTMUNet(
            in_channels=1, out_channels=NUM_CLASS,
            pretrained=False, deep_sup=True, multiscale_att=True,
        ).to(DEVICE)
        m.eval()
        models.append(m)
    return models


def check_shape(name, tensor, expected):
    assert tensor.shape == expected, (
        f"[FAIL] {name}: expected {expected}, got {tensor.shape}"
    )
    print(f"  [OK] {name}: {tuple(tensor.shape)}")


def test_model_forward():
    print("\n--- 1. Plain forward pass ---")
    from models.reslstmunet import ResLSTMUNet

    model = ResLSTMUNet(
        in_channels=1, out_channels=NUM_CLASS,
        pretrained=False, deep_sup=True, multiscale_att=True,
    ).to(DEVICE)
    model.eval()

    x = make_dummy_input()
    with torch.no_grad():
        pred_serial, *aux = model(x)

    assert len(pred_serial) == T, f"Expected {T} timestep outputs, got {len(pred_serial)}"
    check_shape("pred[0]", pred_serial[0], (BATCH, NUM_CLASS, H, W))
    print("  [OK] deep_sup aux outputs:", len(aux))


def test_tta():
    print("\n--- 2. TTA ---")
    from models.reslstmunet import ResLSTMUNet
    from uncertainty import tta_predict

    model = ResLSTMUNet(
        in_channels=1, out_channels=NUM_CLASS,
        pretrained=False, deep_sup=True, multiscale_att=True,
    ).to(DEVICE)
    model.eval()

    x = make_dummy_input()
    mean_probs, uncertainty = tta_predict(model, x, n_transforms=N_TTA)

    assert len(mean_probs) == T
    for t in range(T):
        check_shape(f"mean_probs[{t}]",  mean_probs[t],  (BATCH, NUM_CLASS, H, W))
        check_shape(f"uncertainty[{t}]", uncertainty[t], (BATCH, H, W))

    prob_sum = mean_probs[0].sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "Probabilities do not sum to 1"
    print("  [OK] probabilities sum to 1")


def test_ensemble():
    print("\n--- 3. Checkpoint Ensemble ---")
    from uncertainty import ensemble_predict

    models = make_dummy_models(N_MODELS)
    x = make_dummy_input()
    mean_probs, uncertainty = ensemble_predict(models, x)

    assert len(mean_probs) == T
    for t in range(T):
        check_shape(f"mean_probs[{t}]",  mean_probs[t],  (BATCH, NUM_CLASS, H, W))
        check_shape(f"uncertainty[{t}]", uncertainty[t], (BATCH, H, W))

    prob_sum = mean_probs[0].sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "Probabilities do not sum to 1"
    print("  [OK] probabilities sum to 1")

    # Different models (random weights) should produce different predictions
    assert (uncertainty[0] > 0).any(), \
        "[FAIL] Uncertainty is zero — all checkpoints produced identical outputs"
    print("  [OK] uncertainty is non-zero across checkpoints")


def test_ensemble_tta():
    print("\n--- 4. Combined Ensemble + TTA ---")
    from uncertainty import ensemble_tta_predict

    models = make_dummy_models(N_MODELS)
    x = make_dummy_input()
    mean_probs, uncertainty = ensemble_tta_predict(models, x, n_transforms=N_TTA)

    assert len(mean_probs) == T
    for t in range(T):
        check_shape(f"mean_probs[{t}]",  mean_probs[t],  (BATCH, NUM_CLASS, H, W))
        check_shape(f"uncertainty[{t}]", uncertainty[t], (BATCH, H, W))

    prob_sum = mean_probs[0].sum(dim=1)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5), \
        "Probabilities do not sum to 1"
    print("  [OK] probabilities sum to 1")

    assert (uncertainty[0] > 0).any(), \
        "[FAIL] Uncertainty is zero across all ensemble+TTA combinations"
    print("  [OK] uncertainty is non-zero")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    try:
        test_model_forward()
        test_tta()
        test_ensemble()
        test_ensemble_tta()
        print("\n=== All tests passed ===")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
