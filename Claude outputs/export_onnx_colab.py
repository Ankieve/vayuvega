# Paste into ONE Colab cell (CPU runtime is fine) and run.
# It converts the final model.pth to model.onnx (no PyTorch needed to run it later),
# checks the two give the same answers, then downloads model.onnx.
!pip -q install timm onnx onnxruntime

import numpy as np, torch, timm, onnxruntime as ort
from google.colab import files

print("Upload the FINAL model.pth (the same one used in the app)")
up = files.upload()
pth = [n for n in up if n.endswith(".pth")][0]

# Same loading as backend/predict.py (no ImageNet normalization anywhere)
m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
m.load_state_dict(torch.load(pth, map_location="cpu"))
m.eval()

dummy = torch.rand(1, 3, 224, 224)
torch.onnx.export(
    m, dummy, "model.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=17, dynamo=False,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
)

# Check: PyTorch vs ONNX on 20 random inputs must agree closely
sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
worst = 0.0
for _ in range(20):
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        a = torch.softmax(m(x), 1).numpy()
    b = sess.run(None, {"input": x.numpy()})[0]
    b = np.exp(b - b.max()) / np.exp(b - b.max()).sum()
    worst = max(worst, float(np.abs(a - b).max()))
print("Largest probability difference:", worst)
print("OK - safe to use" if worst < 1e-3 else "PROBLEM - do not use, tell Claude")

import os
print("model.onnx size (MB):", round(os.path.getsize("model.onnx") / 1e6, 1))
files.download("model.onnx")
