---
title: VAYUVEGA Cyclone Intensity
emoji: 🌀
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# VAYUVEGA (SIH26070) - Tropical cyclone intensity dashboard

Upload an infrared satellite image of a tropical cyclone and get the predicted intensity class
(5 IMD classes) with confidence and class probabilities from an EfficientNet-B0 model.

- Model output (REAL): intensity class, confidence, class probabilities.
- Track outlook, wind outlook, pressure, risk index and rapid-intensification flag are SIMULATED
  (historical-analog method and simple rules), and are labelled as such in the app.
- About 60% accuracy on 372 unseen TCIR test images (macro-F1 0.49). Prototype only; not for
  operational forecasting or warnings.
