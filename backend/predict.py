"""
SIH26070 - Cyclone Intensity Prediction
v3 - wind-speed regression ENSEMBLE (model_v3b + model_v4_seed1),
replacing the v2 direct-classification model.pth.

Why this replaced v2
---------------------
v2 (5-class classifier, single model) was evaluated two different ways during
this project:
  - Its own original 372-image held-out test: 60.5% exact.
  - A storm-wise (grouped by storm_id) held-out test of 12 unseen storms,
    520 images, built later specifically to catch leakage: 54.8% exact, but
    97.2% "train" exact on that same split - a 42.4-point train-test gap.
    That gap is the signature of overfitting: earlier train/test splits let
    images from the same storm appear on both sides, so v2 had partly
    memorised storms rather than learned to generalise. See CLAUDE.md and
    MODAK.ipynb for the full diagnosis.

The v3b + v4_seed1 ensemble was trained on the SAME TCIR dataset (no new
MOSDAC data is in it yet - that pipeline is still being built, see
backend/predict_insat_tools.py and prism_tc_build_dataset.py) but with a
storm-wise split enforced from the start, a wind-speed regression target
instead of a direct class label, and 4-rotation test-time averaging.
Measured on the same 12-storm/520-image storm-wise test set:
    exact 53.7%, within-one-class 93.1%, macro-F1 0.487,
    MAE 10.2 kt, train-test gap 6.0 points (vs v2's 42.4).

Honest trade-off - read before presenting this as a strict improvement
------------------------------------------------------------------------
On a live validation against real MOSDAC INSAT-3DR imagery of Cyclone
Biparjoy (36 frames, IBTrACS ground truth), v2 actually scored HIGHER exact
accuracy than the ensemble (v2 83.3% vs ensemble 75.0%), though the ensemble
won on within-one-class (100% vs 94.4%) and adds an MAE figure v2 cannot
produce (4.9 kt). v2's Biparjoy number may partly reflect the same
overfitting the storm-wise test exposed - Biparjoy-like conditions may
simply be well represented in what v2 memorised - so it is not a reason to
prefer v2 for new, unseen storms. But it means the honest claim is "more
consistent across many unseen storms, not uniformly more accurate on every
one," not "strictly better." See MODAK.ipynb's final comparison cells.

Inference backend: ONNX Runtime (torch-free)
---------------------------------------------
The ensemble weights are served from backend/model_v3b.onnx and
backend/model_v4_seed1.pth-converted model_v4_seed1.onnx via onnxruntime-cpu
- PyTorch is NOT needed to run predictions (it is only needed once, offline,
to re-export the .onnx files if the checkpoints ever change). Same weights,
same ImageNet normalization, same 4-rotation averaging, same wind-to-class
mapping - verified head-to-head against the torch path on all 10 bundled
TCIR samples x 4 rotations: max wind disagreement 0.0002 kt, zero class or
confidence changes. If onnxruntime or the .onnx files are missing, this
module falls back to the original torch/timm path (backend/*.pth); if
neither backend loads, importing this module raises, which server.py's
Predictor catches and serves clearly labelled DEMO MODE placeholders for.

Known trade-off of the ONNX path: backend/gradcam.py needs a real torch
module object (`from predict import _model`), so Grad-CAM is unavailable
when the torch backend is not active - server.py degrades gracefully to a
warning (same pattern as the satellite import guard) and the dashboard
hides the Grad-CAM checkbox.

IMPORTANT input pipeline (unchanged): the v3b/v4_seed1 checkpoints WERE
trained with standard ImageNet normalization (unlike v2, which was
explicitly trained WITHOUT it). Do not remove the normalization below;
that would silently hurt accuracy.

Usage in server.py (unchanged contract):
    from predict import predict
    category, confidence, all_probs = predict(uploaded_image)   # PIL.Image

Confidence here is NOT a softmax probability (the model has one regression
output, not 5 class logits). It is derived by treating the model's wind
estimate as the mean of a Normal distribution whose spread (sigma) is
max(the model's own measured test MAE of 10.2 kt, this image's 4-rotation
disagreement) and integrating that distribution over each class's IMD wind
boundaries. This is a documented derivation, not a measured calibration
curve - no Expected Calibration Error has been computed for it. Do not
describe it to judges as "the model's confidence" without this caveat; see
CLAUDE.md.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

from wind_math import (
    CLASSES,
    MEASURED_MAE_KT,
    class_probs_from_wind,
    wind_to_class,
)

# Must stay IDENTICAL (same strings, same order) to backend/logic.py's
# CLASS_ORDER - server.py does logic.CLASS_ORDER.index(category) and
# {c: probs[c] for c in logic.CLASS_ORDER} on whatever predict() returns, so
# any mismatch here throws on every single prediction request. (The
# teammate's original new predict.py used 'Depression/Deep Depression' for
# class 0, which does NOT match logic.CLASS_ORDER's 'Depression' - fixed in
# wind_math.py rather than carried over. See backend/tests/test_wind_math.py,
# which asserts CLASSES == logic.CLASS_ORDER directly.)

BACKEND_DIR = Path(__file__).parent
_ONNX_PATHS = [BACKEND_DIR / "model_v3b.onnx", BACKEND_DIR / "model_v4_seed1.onnx"]
_PTH_PATHS = [BACKEND_DIR / "model_v3b.pth", BACKEND_DIR / "model_v4_seed1.pth"]

# Number of test-time-augmentation rotations per prediction. The measured
# 53.7% exact / 93.1% within-one-class numbers (CLAUDE.md, model_card.py,
# MODAK.ipynb cell 21) were all measured with the default of 4. Lower this
# ONLY as a deployment-specific speed/memory workaround (e.g. a slow free-tier
# CPU host timing out mid-request) via the TTA_ROTATIONS env var - do not
# change the default here, or the live site's real accuracy will quietly
# drift from the numbers the dashboard/model card claim. A deployment running
# fewer than 4 rotations is measurably less accurate than the tested figures
# (TTA rotation-averaging is part of what was measured) and should say so
# wherever those numbers are shown, rather than presenting them as-is.
TTA_ROTATIONS = int(os.environ.get("TTA_ROTATIONS", "4"))

# ImageNet normalization constants - part of the trained pipeline (see
# module docstring). Applied in numpy on the ONNX path, via
# torchvision.transforms on the torch fallback path; both are the same math.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Exposed for diagnostics (which engine is actually serving). Not part of
# any API response schema.
BACKEND = None  # "onnx" | "torch" once loaded


def _load_onnx():
    """Load the ensemble via onnxruntime. Raises on any problem (missing
    package, missing files, bad graph) so the caller can try torch next."""
    import onnxruntime as ort

    missing = [p for p in _ONNX_PATHS if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing ONNX weights: " + ", ".join(p.name for p in missing))
    opts = ort.SessionOptions()
    # Be a good neighbour on tiny shared hosts: no thread-pool fan-out for
    # a single small inference; results are deterministic either way.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sessions = [
        ort.InferenceSession(str(p), sess_options=opts,
                             providers=["CPUExecutionProvider"])
        for p in _ONNX_PATHS
    ]
    # Sanity: one input (NCHW float32), one scalar wind output.
    for sess, p in zip(sessions, _ONNX_PATHS):
        inp = sess.get_inputs()[0]
        if list(inp.shape[1:]) != [3, 224, 224]:
            raise ValueError(f"{p.name}: unexpected input shape {inp.shape}")
    return sessions


def _load_torch():
    """Original torch/timm backend - fallback when ONNX is unavailable."""
    import torch
    import timm
    from torchvision import transforms

    class _WindEnsemble(torch.nn.Module):
        """Averages the wind output of the ensemble members."""

        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, x):
            return torch.stack([m(x) for m in self.members]).mean(0)

    def _load_member(path):
        sd = torch.load(str(path), map_location=torch.device("cpu"))
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        n_out = sd["classifier.weight"].shape[0]
        if n_out != 1:
            raise ValueError(
                f"{path.name}: expected a 1-output wind-regression checkpoint, "
                f"got {n_out} outputs - wrong file?")
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
        m.load_state_dict(sd)
        m.eval()
        return m

    members = [_load_member(p) for p in _PTH_PATHS]
    ensemble = _WindEnsemble(members)
    ensemble.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Representative member for backend/gradcam.py (single concrete conv
    # architecture - see module docstring for what this simplification means).
    return members, ensemble, transform, members[0]


# Loaded once at import time. ONNX first (no torch needed); torch fallback
# second; a raise here is caught by server.py's Predictor, which serves
# clearly labelled DEMO MODE placeholders.
try:
    _sessions = _load_onnx()
    BACKEND = "onnx"
except Exception as _onnx_exc:  # noqa: BLE001
    try:
        _torch_members, _torch_ensemble, _torch_transform, _model = _load_torch()
        BACKEND = "torch"
    except Exception as _torch_exc:  # noqa: BLE001
        raise RuntimeError(
            f"No inference backend available (onnx: {_onnx_exc}; "
            f"torch: {_torch_exc})")


def _preprocess_numpy(image):
    """torchvision-equivalent Resize(224) + ToTensor + ImageNet Normalize,
    in numpy. Resize uses BILINEAR, matching torchvision's default; verified
    against the torch path on all bundled samples (max wind diff 0.0002 kt).
    Returns float32 NCHW with a batch dim."""
    arr = np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR),
                      dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def predict(image):
    """
    image: a PIL.Image (any mode - converted to RGB internally).
    Returns: (category: str, confidence: float as %, all_probs: dict of all
    5 classes -> %) - identical shape to the old v2 predict(), so
    server.py's Predictor needs no changes.
    """
    image = image.convert("RGB")
    if BACKEND == "onnx":
        base = _preprocess_numpy(image)
        winds = np.array([
            float(np.mean([s.run(None, {"input": np.rot90(base, k, (2, 3)).copy()})[0][0, 0]
                           for s in _sessions]))
            for k in range(TTA_ROTATIONS)
        ]) * 100.0
    else:
        import torch  # noqa: F401  (torch backend already proven importable)
        tensor = _torch_transform(image).unsqueeze(0)
        with torch.no_grad():
            # Test-time-averaging across TTA_ROTATIONS rotations (default 4,
            # see the module-level comment above) - cyclones look the same
            # rotated, so disagreement across rotations is a real per-image
            # uncertainty signal (see MEASURED_MAE_KT above for why it's floored).
            winds = np.array([
                float(_torch_ensemble(torch.rot90(tensor, k, (2, 3)))[0, 0]) * 100.0
                for k in range(TTA_ROTATIONS)
            ])
    kt = float(winds.mean())
    spread = float(winds.max() - winds.min())
    sigma = max(MEASURED_MAE_KT, spread)

    idx = wind_to_class(kt)
    category = CLASSES[idx]
    probs = class_probs_from_wind(kt, sigma)
    all_probs = {c: round(p * 100.0, 2) for c, p in zip(CLASSES, probs)}
    confidence = all_probs[category]
    return category, confidence, all_probs
