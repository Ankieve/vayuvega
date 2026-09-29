"""
Model Transparency panel data (SIH26070 / PRISM-TC).

Everything in MODEL_CARD below is copied from CLAUDE.md and the team's own
test run of the shipped classifier - nothing here is invented for the demo.
The goal of this module is to make those numbers and caveats available to
the frontend as structured JSON instead of leaving them only in a markdown
file nobody in the room will open. If a number below is ever updated after
retraining, update it here AND in CLAUDE.md's "Pitch numbers" section so the
two never drift apart.

This endpoint (GET /api/model/transparency) reports facts about the model
that was trained. It is intentionally static/backend-only: it does not
change per-request the way /api/predict does.
"""

MODEL_CARD = {
    "model": {
        "architecture": "EfficientNet-B0 (via timm), 5-class classification head",
        "input": "Single infrared satellite image chip, resized to 224x224, RGB "
                 "tensor with NO ImageNet normalization (the checkpoint was "
                 "trained without it - adding it drops accuracy from ~60% to ~30%).",
        "training_data": "TCIR dataset only (pre-cropped, storm-centred IR chips). "
                          "Labels come from IBTrACS-derived intensity categories.",
    },

    "classes": [
        "Depression",
        "Cyclonic Storm",
        "Severe Cyclonic Storm",
        "Very Severe Cyclonic Storm",
        "Extremely Severe/Super Cyclone",
    ],

    "metrics": {
        "test_set_size": 372,
        "accuracy_pct": 60.0,
        "macro_f1": 0.49,
        "average_error_severity_levels": 0.51,
        "expected_calibration_error_pct": 27.0,
        "note": "Measured once on a held-out TCIR test split. Not re-measured "
                "against live satellite imagery (see 'known_failure_modes' below).",
    },

    # What is actually known about class balance in the 372-image test set.
    # Only the counts that were explicitly recorded are listed - the rest are
    # left out rather than guessed, in keeping with this project's honesty rules.
    "test_set_class_notes": [
        {"class": "Depression", "share_pct": 50.0,
         "note": "Half the test set. Always predicting Depression would score 50% accuracy on its own."},
        {"class": "Very Severe Cyclonic Storm", "count": 10,
         "note": "Too few examples to reliably measure performance on a class this rare."},
        {"class": "Extremely Severe/Super Cyclone", "count": 6,
         "note": "Even fewer examples than Very Severe - treat any number for this class with real caution."},
    ],

    "known_failure_modes": [
        "Confidence is overconfident: Expected Calibration Error is about 27%, so a "
        "reported '92% confidence' does not mean the model is right 92% of the time.",
        "Single infrared channel only - no visible, water-vapour, or microwave channel, "
        "unlike operational Dvorak-style intensity estimation.",
        "The two most severe classes (Very Severe, Extremely Severe) have only 10 and 6 "
        "test examples respectively, so accuracy on the classes that matter most is the "
        "least statistically reliable part of the reported number.",
        "Trained and evaluated only on TCIR chips (pre-cropped, storm-centred). Never "
        "measured against live wide-area satellite imagery like the near-real-time "
        "IMD/INSAT feed on this dashboard - that is exactly why live-image inference is "
        "disabled by default.",
        "Does not locate or detect a storm inside a raw satellite frame - it only "
        "classifies an image someone has already selected or cropped.",
        "No temporal/sequence model: each classification is a single independent image, "
        "not a track of a storm's evolution over time.",
    ],

    "honesty_notes": [
        "Track, wind outlook, pressure, risk index and the analog-based "
        "rapid-intensification flag are historical-analog and rule-based methods, "
        "not machine-learning forecasts - see backend/logic.py.",
        "SST and wind shear now feed a separate, clearly-labelled rule-based "
        "environment heuristic (not the AI classifier) - see backend/logic.py's "
        "environment_favors_intensification(). Humidity and vorticity remain "
        "accepted-but-unused, reserved for a possible future ERA5 fusion.",
    ],
}


def get_model_card():
    """Return a fresh copy (so callers can't mutate the shared dict) plus a
    live 'calibration' section - the one part of this card that isn't purely
    static, since whether temperature scaling has actually been fitted
    depends on whether backend/fit_calibration.py has been run on this
    machine (see calibration.py). Importing calibration here needs no torch,
    so this stays true even before anyone has fitted anything."""
    import copy

    from calibration import load_temperature

    card = copy.deepcopy(MODEL_CARD)
    temperature, fitted = load_temperature()
    if fitted:
        card["calibration"] = {
            "fitted": True,
            "temperature": temperature,
            "note": "Confidence scores are divided by this learned temperature before the "
                    "softmax (temperature scaling) - this spreads out overconfident "
                    "probabilities without ever changing the predicted class. See "
                    "backend/temperature.json for how and when it was fitted.",
        }
    else:
        card["calibration"] = {
            "fitted": False,
            "temperature": 1.0,
            "note": "No fitted temperature yet - confidence scores are NOT calibrated "
                    "(this is the same ~27% ECE overconfidence documented above). Run "
                    "backend/fit_calibration.py on a machine with the real model.pth and "
                    "torch installed to fit one; predict.py will pick it up automatically.",
        }
    return card
