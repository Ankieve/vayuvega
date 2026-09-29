# PRISM-TC / VAYUVEGA — Honest Gap Analysis vs. SIH26070 Problem Statement

**Problem statement (verbatim):** *"To develop an Artificial Intelligence (AI) / Machine
Learning (ML) based system for identification, classification, and prediction of different
tropical cyclone patterns using multi-source satellite data."*

This document maps the current system honestly against each part of that statement:
identification, classification, prediction, and multi-source satellite data. It is written to
be read before the presentation, not during it — so none of this lands as a surprise if a judge
digs in.

---

## 1. Identification — the weakest of the three verbs

The system does **not** actually locate or detect a cyclone within a satellite image.

- TCIR, the dataset the model was trained on, ships pre-cropped, storm-centred chips — someone
  else already did the "find the storm" work when TCIR was built.
- The pipeline only ever classifies an image a human has already selected (a sample) or
  uploaded (already cropped to the storm).
- Given the live wide-area INSAT image, the system has no way to find where a cyclone is inside
  that frame — this is exactly why live inference is disabled by default
  (`ENABLE_LIVE_INFERENCE=false`).

**If asked:** "Show me it finding a cyclone in a raw satellite pass" — there is nothing to show.
This capability does not exist yet.

---

## 2. Classification — real, but narrower than it sounds

This is the part that actually works, with real caveats.

| Metric | Value | Caveat |
|---|---|---|
| Accuracy | ~60% | On 372 held-out TCIR test images |
| Macro-F1 | 0.49 | Test set is 50% "Depression" — always guessing Depression scores 50% accuracy |
| Very Severe examples in test set | 10 | Too few to reliably measure performance on the class that matters most |
| Extremely Severe examples in test set | 6 | Same issue, even more extreme |
| Expected Calibration Error (ECE) | ~27% | Confidence scores are meaningfully overconfident — "92% confidence" does not mean 92% correct |

Additional structural limitation: classification uses **a single infrared channel only** — no
visible, no water vapour, no microwave. Real operational intensity estimation (Dvorak and its
successors) leans heavily on multi-channel and microwave structure precisely because IR alone is
ambiguous about a storm's inner core.

---

## 3. Prediction — simulated, not learned; the biggest gap against the brief

The track outlook, wind outlook, pressure estimate, risk index, and rapid-intensification flag
are **explicitly not machine learning**.

- `backend/logic.py`'s `analog_outlook()` finds the closest historical IBTrACS storm by
  class/position and copies its future track and wind trend forward — a nearest-neighbour
  analog lookup, not a trained model.
- There is no trained temporal/sequence model (no CNN+GRU — an earlier draft claimed this and it
  was deliberately removed for honesty; see `CLAUDE.md`'s "Honesty rules").

**If a judge reads "prediction of tropical cyclone patterns" as "forecast how this storm evolves
using ML,"** the honest answer is: it doesn't. It pattern-matches history and extrapolates with
simple rules. This is a disclosed, deliberate design choice — but it is the single biggest
distance between what the problem statement asks for and what is built.

---

## 4. Multi-source satellite data — the most direct miss against the problem statement's wording

Both the trained model and the live feed use **exactly one data source at a time**, never a
fusion of sources.

- **Training:** TCIR only — a single-channel product derived from a mix of historical
  geostationary satellites, delivered as one pre-processed channel. No channel selection or
  fusion available at training time.
- **Live feed:** IMD/INSAT-3D IR1 only, as of the latest version (previously an unverified
  Himawari attempt — see `backend/satellite/README.md` section 0 for that history).
- **No fusion of:** infrared + visible + water vapour + microwave; geostationary + polar-orbiting
  passes; multiple agencies' products.
- **No non-satellite data either.** The `/api/predict` endpoint literally accepts `sst`,
  `wind_shear`, `humidity`, and `vorticity` as input fields but **silently ignores all four** —
  `CLAUDE.md` states this explicitly: *"accepted but NOT used yet"* (planned ERA5 fusion, never
  implemented).

**If asked:** send wildly different SST or wind-shear values through the API and the prediction
will not change at all. This is an easy thing for a judge to catch — have the honest explanation
ready (see above) rather than being caught by it live.

---

## 5. The live satellite card and the AI model are two disconnected pipelines

Worth naming out loud before the demo, since it is easy to misread as one integrated system:

- The live satellite card shows a real, live INSAT image updating every few minutes — genuinely
  a working feature on its own.
- By design, it does **not** feed the classifier, because the model was never trained on wide
  regional images like that one — only tight, storm-centred TCIR chips.
- So the demo has two separate, honestly-labelled but disconnected halves: *"here's a live
  picture"* and *"here's an AI classification of a picture you pick."*

**If asked:** "So it just classified that live image?" — the answer is no, and the reason is a
genuine train/inference data-mismatch (see `backend/satellite/inference_bridge.py`'s docstring
for the full technical reasoning), not an oversight.

---

## 6. Smaller but real gaps

- **No live validation.** Accuracy numbers come only from the static TCIR test set; there is no
  way in a hackathon setting to compare live classifications against real IMD/JTWC bulletins.
- **No storm tracking over time.** Each classification is a single independent image call, not a
  video/sequence model tracking one storm across time.
- **Distribution shift, unmeasured.** The model's training distribution (historical TCIR
  imagery) differs radiometrically from live INSAT quicklooks. Even if live inference were
  enabled, its accuracy on that different-looking imagery has never been measured — it would be
  unvalidated, not just disclosed as experimental.
- **Approximate live-image framing.** The IMD source's "Asia sector" extent is read off the
  image's own labelled gridlines, not pixel-calibrated — fine for a dashboard visual, not
  precise enough for scientific geolocation.

---

## 7. What still stands as a genuine strength

For balance — this is not a case for scrapping the project, it's a punch list:

- A real, working, trained classifier with honestly-reported (not inflated) accuracy is more
  credible to most judges than a polished-looking fake.
- The satellite integration is genuinely live, free, and correctly, honestly labelled at every
  failure mode (LIVE/STALE/UPDATING/ERROR/NO DATA) rather than faking data when something breaks.
- Every simulated/rule-based value in the UI is explicitly tagged as such (SIMULATED, EST.,
  PLACEHOLDER) rather than presented as AI output — this kind of self-disclosure is unusual and
  works in the team's favour if a judge notices it.

---

## 8. Two questions worth rehearsing an answer to

1. **"You said multi-source — where's the fusion?"**
   Honest answer: today it's single-source at both training and inference time (TCIR for
   training, IMD/INSAT for the live feed); multi-channel/multi-satellite fusion and the
   already-scaffolded `sst`/`wind_shear`/`humidity`/`vorticity` predictors are documented as
   planned future work, not yet implemented.

2. **"Is 'prediction' here actually machine learning?"**
   Honest answer: no — track/wind/pressure/risk outputs are a historical-analog + rule-based
   method, clearly tagged SIMULATED in the UI; only the intensity **classification** from an
   image is the validated AI/ML component.
