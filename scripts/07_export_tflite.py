#!/usr/bin/env python
"""Methods 2.12 — export the deployment model to TensorFlow Lite (FP32 and INT8 PTQ).

  python scripts/07_export_tflite.py --fold-dir runs/mobilenet_v2__full/fold0 \
      --metadata data/metadata.csv --image-root data/preprocessed --out deploy/

Route: PyTorch -> ONNX -> TensorFlow SavedModel (onnx2tf, NHWC) -> TFLite.
INT8 = full-integer post-training quantisation of weights and activations,
calibrated on `--n-calib` real development images (float32 input/output kept
so the app can pass normalised pixels directly).

Needs: pip install onnx onnxsim "onnx2tf[tensorflow]" tensorflow tf_keras
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from PIL import Image

from adpipe.data import build_transforms, load_metadata
from adpipe.deploy import DeployWrapper, load_checkpoint
from adpipe.models import INPUT_SIZE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", default="deploy")
    ap.add_argument("--n-calib", type=int, default=200)
    ap.add_argument("--onnx-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model, meta_enc, cfg, T = load_checkpoint(a.fold_dir)
    size = INPUT_SIZE[cfg["backbone"]]
    wrapper = DeployWrapper(model, T).eval()
    meta_dim = max(meta_enc.dim, 1)  # keep a (dummy) meta input so the graph signature is fixed
    if meta_enc.dim == 0:
        class NoMeta(torch.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, image, meta):
                return self.w(image, meta[:, :0])
        wrapper = NoMeta(wrapper).eval()

    onnx_path = os.path.join(a.out, "model.onnx")
    dummy = (torch.randn(1, 3, size, size), torch.zeros(1, meta_dim))
    torch.onnx.export(wrapper, dummy, onnx_path, input_names=["image", "meta"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    print(f"ONNX -> {onnx_path}")
    if a.onnx_only:
        return

    # calibration data (NHWC float32, ImageNet-normalised) from real development images
    df = load_metadata(a.metadata)
    dev = df[(df.split == "dev") & (df.synthetic == 0)].sample(frac=1, random_state=0).head(a.n_calib)
    tf_eval = build_transforms(size, train=False)
    imgs = np.stack([tf_eval(Image.open(os.path.join(a.image_root, p)).convert("RGB")).numpy()
                     for p in dev.image_path]).transpose(0, 2, 3, 1).astype(np.float32)
    metas = meta_enc.transform(dev) if meta_enc.dim else np.zeros((len(dev), 1), np.float32)
    np.save(os.path.join(a.out, "calib_images.npy"), imgs)
    np.save(os.path.join(a.out, "calib_meta.npy"), metas)

    sm_dir = os.path.join(a.out, "saved_model")
    subprocess.run(["onnx2tf", "-i", onnx_path, "-o", sm_dir, "-osd", "-tb", "tf_converter"], check=True)

    import tensorflow as tf

    def convert(int8: bool):
        conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
        if int8:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]

            def rep():
                for i in range(len(imgs)):
                    # dict form: matched to the SavedModel signature by input name
                    yield {"image": imgs[i:i + 1], "meta": metas[i:i + 1]}
            conv.representative_dataset = rep
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return conv.convert()

    for name, q in (("model_fp32.tflite", False), ("model_int8.tflite", True)):
        path = os.path.join(a.out, name)
        open(path, "wb").write(convert(q))
        print(f"{name}: {os.path.getsize(path) / 1e6:.2f} MB")
    meta_enc.to_json(os.path.join(a.out, "meta_encoder.json"))
    open(os.path.join(a.out, "temperature.txt"), "w").write(str(T))
    print("Check the input order with scripts/08_benchmark_tflite.py (it matches inputs by shape).")


if __name__ == "__main__":
    main()
