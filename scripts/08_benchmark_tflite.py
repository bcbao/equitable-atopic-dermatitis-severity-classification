#!/usr/bin/env python
"""Methods 2.12 — size, CPU latency and accuracy of FP32 vs INT8 TFLite models.

  python scripts/08_benchmark_tflite.py --deploy-dir deploy --metadata data/metadata.csv \
      --image-root data/preprocessed --threads 4

Latency: batch size 1, 50 warm-up runs, mean over 500 timed runs.
Accuracy: macro-F1 on the held-out test set for both models (Δ = INT8 − FP32).

Android phone: push the .tflite files and use TensorFlow's benchmark_model tool, e.g.
  adb push deploy/model_int8.tflite /data/local/tmp/
  adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/model_int8.tflite \
      --num_threads=4 --warmup_runs=50 --num_runs=500
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from PIL import Image

from adpipe.data import MetaEncoder, build_transforms, load_metadata
from adpipe.metrics import macro_f1


def get_interpreter(path, threads):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def io_index(it):
    ins = it.get_input_details()
    img = next(d for d in ins if len(d["shape"]) == 4)
    meta = next((d for d in ins if len(d["shape"]) == 2), None)
    return img, meta, it.get_output_details()[0]


def run(it, img, meta=None):
    di, dm, do = io_index(it)
    it.set_tensor(di["index"], img)
    if dm is not None:
        it.set_tensor(dm["index"], meta)
    it.invoke()
    return it.get_tensor(do["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-dir", default="deploy")
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    a = ap.parse_args()

    df = load_metadata(a.metadata)
    test = df[df.split == "test"].reset_index(drop=True)
    meta_enc = MetaEncoder.from_json(os.path.join(a.deploy_dir, "meta_encoder.json"))
    metas = meta_enc.transform(test) if meta_enc.dim else np.zeros((len(test), 1), np.float32)

    rows = []
    for name in ("model_fp32.tflite", "model_int8.tflite"):
        path = os.path.join(a.deploy_dir, name)
        it = get_interpreter(path, a.threads)
        di, _, _ = io_index(it)
        size = int(di["shape"][1])
        tf_eval = build_transforms(size, train=False)
        x0 = np.zeros(di["shape"], np.float32)
        m0 = metas[:1] if len(metas) else np.zeros((1, 1), np.float32)
        for _ in range(a.warmup):
            run(it, x0, m0)
        t = []
        for _ in range(a.runs):
            s = time.perf_counter()
            run(it, x0, m0)
            t.append(time.perf_counter() - s)
        probs = []
        for i, p in enumerate(test.image_path):
            x = tf_eval(Image.open(os.path.join(a.image_root, p)).convert("RGB")).numpy()
            probs.append(run(it, x.transpose(1, 2, 0)[None].astype(np.float32), metas[i:i + 1])[0])
        probs = np.array(probs)
        rows.append({"model": name, "size_MB": os.path.getsize(path) / 1e6,
                     "latency_ms_mean": 1e3 * np.mean(t), "latency_ms_sd": 1e3 * np.std(t),
                     "test_macro_f1": macro_f1(test.label, probs.argmax(1)) if len(test) else np.nan})
    res = pd.DataFrame(rows)
    fp, q = res.iloc[0], res.iloc[1]
    print(res.to_string(index=False))
    print(f"\nsize reduction: {fp.size_MB / q.size_MB:.1f}x | latency change: "
          f"{100 * (q.latency_ms_mean - fp.latency_ms_mean) / fp.latency_ms_mean:+.0f}% | "
          f"Δ macro-F1 (INT8 − FP32): {100 * (q.test_macro_f1 - fp.test_macro_f1):+.1f} pts")
    res.to_csv(os.path.join(a.deploy_dir, "benchmark.csv"), index=False)


if __name__ == "__main__":
    main()
