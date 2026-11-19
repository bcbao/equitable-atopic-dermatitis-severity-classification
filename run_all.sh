#!/usr/bin/env bash
# Full pipeline, in the order of the manuscript. Run from the repository root.
set -euo pipefail
CFG=configs/default.yaml

# 0. Table 1 (QC summary)
python scripts/00_table1_qc.py --qc-log data/qc_log.csv --metadata data/metadata.csv --out results/table1_qc.csv

# 1. Preprocessing (2.4)  raw images -> data/preprocessed + data/metadata.csv
python scripts/01_preprocess.py --metadata data/metadata_raw.csv --raw-root data/raw \
       --out-root data/preprocessed --out-csv data/metadata.csv

# 2. Skin-tone-targeted generative augmentation (2.8), CUDA GPU required
#    2a. one LoRA adapter per split on Stable Diffusion v1.5 -> ./lora_fold_1 ... ./lora_fold_5
python scripts/02a_train_lora.py --metadata data/metadata.csv --image-root data/preprocessed --out .
#    2b. img2img generation from Fitzpatrick I-III training images with the split's own LoRA
python scripts/02b_generate_synthetic.py --metadata data/metadata.csv --image-root data/preprocessed --lora-root .
#    -> dermatologist review of data/synthetic_manifest.csv (mark rejected), then:
#    python scripts/02b_generate_synthetic.py --metadata data/metadata.csv --merge-only

# 3. 5-fold CV + held-out test for 3 backbones x baseline/full, plus ablations on MobileNetV2
for BB in vgg16 inception_v3 mobilenet_v2; do
  for V in baseline full; do
    python scripts/03_run_cv.py --config $CFG --backbone $BB --variant $V \
           --set metadata_csv=data/metadata_with_synthetic.csv
  done
done
for V in no_ldam_drw no_style_aug no_metadata no_domain_adapt; do
  python scripts/03_run_cv.py --config $CFG --backbone mobilenet_v2 --variant $V \
         --set metadata_csv=data/metadata_with_synthetic.csv
done

# 4. Metrics, CIs, fairness, calibration, statistical tests
python scripts/04_evaluate.py --runs runs --out results --compare \
  mobilenet_v2__full:mobilenet_v2__baseline \
  mobilenet_v2__full:mobilenet_v2__no_ldam_drw \
  mobilenet_v2__full:mobilenet_v2__no_style_aug \
  mobilenet_v2__full:mobilenet_v2__no_metadata \
  mobilenet_v2__full:mobilenet_v2__no_domain_adapt \
  vgg16__full:vgg16__baseline inception_v3__full:inception_v3__baseline

# 5. Figure 3
python scripts/05_figure3.py --results results --runs runs --out figures/figure3.png

# 6. Figure 2 (choose 4 example images yourself)
# python scripts/06_figure2_topographic.py --images A.png B.png C.png D.png \
#        --predictions results/test_with_adtss__mobilenet_v2__full.csv --out figures/figure2.png

# 7-8. INT8 TFLite export + benchmark (2.12)
python scripts/07_export_tflite.py --fold-dir runs/mobilenet_v2__full/refit \
       --metadata data/metadata.csv --image-root data/preprocessed --out deploy
python scripts/08_benchmark_tflite.py --deploy-dir deploy --metadata data/metadata.csv --image-root data/preprocessed
