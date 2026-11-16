# AD skin-tone pipeline

Python/PyTorch implementation of the methods described in
*"Reducing Skin-Tone Bias in AI-Based Atopic Dermatitis Diagnosis: Class-Balanced Transfer Learning with Generative Data Augmentation"*.

The pipeline covers preprocessing, patient-level cross-validation, three ImageNet-pretrained backbones (VGG16, InceptionV3, MobileNetV2), class-imbalance handling (LDAM-DRW with class-balanced sampling), skin-tone-targeted style-transfer augmentation for Fitzpatrick IV–VI, clinical metadata fusion, domain-adversarial adaptation, calibration and fairness evaluation, the AD Topographic Severity Score (AD-TSS), and INT8 TensorFlow Lite export for edge deployment.

> **Data availability.** Patient images and clinical metadata are not included in this repository. Only code and a metadata template are provided.
>
> **Note.** This code implements the procedures described in the Methods section. Re-running it may give results that differ slightly from the published values because of random seeds, fold assignment, hardware and library versions.

## Repository structure

| File | Manuscript section |
|---|---|
| `src/adpipe/preprocess.py` | 2.4 Shades-of-Gray colour constancy, DullRazor artifact removal, ROI cropping |
| `src/adpipe/data.py` | 2.3, 2.5, 2.9 Metadata loading, patient-level folds, clinical variable encoding, class-balanced sampler |
| `src/adpipe/models.py` | 2.6, 2.9, 2.10 VGG16 / InceptionV3 / MobileNetV2, late metadata fusion, domain-adversarial head (GRL) |
| `src/adpipe/losses.py` | 2.7 LDAM loss with deferred re-weighting (DRW) |
| `src/adpipe/train.py` | 2.6–2.10 Training of one fold (10 frozen epochs → fine-tuning up to 60 epochs, early stopping) |
| `src/adpipe/style_augment.py` | 2.8 Neural style transfer to Fitzpatrick IV–VI skin tone with morphology check |
| `src/adpipe/metrics.py` | 2.11 Macro-F1, AUROC, ECE, Brier score, NRI, temperature scaling, fairness gaps |
| `src/adpipe/stats.py` | 2.13 Patient-level bootstrap, paired bootstrap, DeLong test, McNemar test |
| `src/adpipe/adtss.py` | AD-TSS (Fig. 2E) |
| `src/adpipe/topographic.py` | 3D topographic key-point maps (Fig. 2A–D) |
| `src/adpipe/deploy.py` | 2.12 Loading trained checkpoints for export |
| `scripts/00_table1_qc.py` | Table 1 (quality-control summary) |
| `scripts/01…08_*.py` | Pipeline steps in order (see `run_all.sh`) |
| `configs/default.yaml` | All hyperparameters and the baseline / full / ablation variants |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
A CUDA GPU is recommended. The manuscript used Python 3.10 and PyTorch 2.1; the code has also been tested with newer PyTorch 2.x releases.

## Data preparation

1. Put the raw images in `data/raw/`.
2. Create `data/metadata_raw.csv` following `data/metadata_template.csv`, one row per image:
   - `image_path`, `patient_id`, `severity` (mild / moderate / severe), `fitzpatrick` (1–6)
   - `split`: `dev` (development set) or `test` (held-out set from a different site/devices)
   - `site`, `device`: acquisition site and device, used for domain adaptation
   - `age`, `sex`, `anatomic_site`, `disease_duration_months`, `atopic_history` (leave blank if missing)
   - `phenotype` (optional): xerotic / erythematous / excoriated / lichenified / exudative — only needed for Fig. 3D
   - `roi_x, roi_y, roi_w, roi_h` (optional): manual ROI box; if blank, the ROI is detected automatically
3. `data/qc_log.csv` (for Table 1): one row per collected image with `excluded` (0/1) and `exclusion_reason`.

## Usage

Run the whole pipeline with `bash run_all.sh`, or step by step:

```bash
python scripts/01_preprocess.py --metadata data/metadata_raw.csv --raw-root data/raw
python scripts/02_style_augment.py --metadata data/metadata.csv --method nst
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant full --set metadata_csv=data/metadata_with_synthetic.csv
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant baseline
python scripts/04_evaluate.py --compare mobilenet_v2__full:mobilenet_v2__baseline
python scripts/05_figure3.py
```
Any configuration value can be overridden from the command line, e.g. `--set lr=3e-4 batch_size=16`.

Outputs:
- `runs/<backbone>__<variant>/` — fold models, out-of-fold predictions (`cv_predictions.csv`) and held-out test predictions (`test_predictions.csv`)
- `results/cv_summary.csv` — mean ± SD across the 5 folds
- `results/test_summary.csv` — held-out test metrics with 95% CIs (1,000 patient-level bootstrap resamples)
- `results/comparisons.csv` — paired bootstrap, DeLong, McNemar, NRI and Fitzpatrick-stratified differences
- `results/test_with_adtss__*.csv` — per-image AD-TSS

## Implementation choices

Some details are not fully specified in the manuscript. The defaults below can be changed in `configs/default.yaml`.

| Item | Choice in this code | Where to change |
|---|---|---|
| Data used for early stopping | 15% of patients held out *inside* each training fold (inner validation) for early stopping and temperature scaling; the outer validation fold is untouched | `inner_val_frac` (0 = use the outer fold) |
| Held-out test predictions | Average of the probabilities of the 5 fold models | `test_mode: refit` = retrain on the full development set |
| DRW "after epoch 20" | Counted over all epochs, including the 10 frozen epochs | `drw_start_epoch` |
| Balanced sampling and DRW | Both enabled, as described | `balanced_sampler`, `drw` |
| NRI definition | Multiclass categorical NRI: for each true class, (proportion moved wrong→right) − (proportion moved right→wrong), averaged over the 3 classes | `metrics.nri_multiclass` |
| AD-TSS cut-offs | Score rounded to 2 decimals, then ≤ 0.33 / ≤ 0.66 / above | `adtss.py` |
| Fig. 3D predicted phenotype | Auxiliary morphology head; predicted bin = predicted severity + the more probable of that severity's two phenotypes | `aux_phenotype`, `phenotype_bins` |
| Style references for augmentation | Preferably a folder of normal-skin patches from Fitzpatrick IV–VI patients (`--style-dir`); otherwise IV–VI development images | `02_style_augment.py` |
| Fitzpatrick type in metadata fusion | Included as an input variable (Section 2.9) | `meta_categorical` |

## Quick test without real data

```bash
python tests/make_dummy_data.py dummy 40
python scripts/01_preprocess.py --metadata dummy/metadata_raw.csv --raw-root dummy/raw --out-root dummy/pre --out-csv dummy/metadata.csv
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant full \
  --set metadata_csv=dummy/metadata.csv image_root=dummy/pre pretrained=false freeze_epochs=1 finetune_epochs=1 num_workers=0
```
The dummy images only check that the code runs end-to-end; the resulting metrics are meaningless.
