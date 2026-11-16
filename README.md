# AD skin-tone pipeline — code viết lại theo bài báo

Code Python/PyTorch viết lại theo phần **Methods** của bài
*"Reducing Skin-Tone Bias in AI-Based Atopic Dermatitis Diagnosis: Class-Balanced Transfer Learning with Generative Data Augmentation"*.

> Đây là bản **dựng lại từ mô tả trong bài**, không phải code gốc. Chạy lại trên dữ liệu thật sẽ cho số liệu gần nhưng **không trùng khớp tuyệt đối** với số trong bản thảo (seed, chia fold, chi tiết không ghi trong Methods). Nếu số mới khác, cần cập nhật lại số trong bài.

## Cấu trúc

| File | Mục trong bài |
|---|---|
| `src/adpipe/preprocess.py` | 2.4 Shades-of-Gray, DullRazor, cắt ROI |
| `src/adpipe/data.py` | 2.3, 2.5, 2.9 metadata, chia fold theo bệnh nhân, mã hoá biến lâm sàng, sampler cân bằng lớp |
| `src/adpipe/models.py` | 2.6, 2.9, 2.10 VGG16 / InceptionV3 / MobileNetV2, late fusion, nhánh domain-adversarial (GRL) |
| `src/adpipe/losses.py` | 2.7 LDAM-DRW |
| `src/adpipe/train.py` | 2.6–2.10 huấn luyện 1 fold (đóng băng 10 epoch → fine-tune ≤60 epoch, early stopping) |
| `src/adpipe/style_augment.py` | 2.8 neural style transfer sang tông da Fitzpatrick IV–VI + kiểm tra hình thái |
| `src/adpipe/metrics.py` | 2.11 macro-F1, AUROC, ECE, Brier, NRI, temperature scaling, khoảng cách công bằng |
| `src/adpipe/stats.py` | 2.13 bootstrap theo bệnh nhân, paired bootstrap, DeLong, McNemar |
| `src/adpipe/adtss.py` | AD-TSS (Hình 2E) |
| `src/adpipe/topographic.py` | Bản đồ topographic 3D (Hình 2A–D) |
| `scripts/00_table1_qc.py` | Bảng 1 |
| `scripts/01…08_*.py` | Các bước chạy theo thứ tự (xem `run_all.sh`) |
| `configs/default.yaml` | Toàn bộ siêu tham số + các biến thể baseline / full / ablation |

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Nên chạy trên GPU (CUDA). Bài ghi Python 3.10 + PyTorch 2.1; code đã chạy thử với PyTorch 2.x mới hơn.

## Chuẩn bị dữ liệu

1. Ảnh gốc để trong `data/raw/`.
2. Tạo `data/metadata_raw.csv` theo mẫu `data/metadata_template.csv`, mỗi dòng 1 ảnh:
   - `image_path`, `patient_id`, `severity` (mild/moderate/severe), `fitzpatrick` (1–6)
   - `split`: `dev` (782 ảnh) hoặc `test` (142 ảnh, cơ sở/thiết bị khác)
   - `site`, `device`: dùng cho domain adaptation
   - `age`, `sex`, `anatomic_site`, `disease_duration_months`, `atopic_history` (để trống nếu thiếu)
   - `phenotype` (tuỳ chọn): xerotic / erythematous / excoriated / lichenified / exudative — chỉ cần cho Hình 3D
   - `roi_x, roi_y, roi_w, roi_h` (tuỳ chọn): khung ROI vẽ tay; nếu trống sẽ tự tìm ROI
3. `data/qc_log.csv` (cho Bảng 1): mỗi ảnh đã thu thập (1.036 ảnh) với `excluded` (0/1) và `exclusion_reason`.

## Chạy

Chạy toàn bộ: `bash run_all.sh`. Hoặc từng bước:

```bash
python scripts/01_preprocess.py --metadata data/metadata_raw.csv --raw-root data/raw
python scripts/02_style_augment.py --metadata data/metadata.csv --method nst
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant full --set metadata_csv=data/metadata_with_synthetic.csv
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant baseline
python scripts/04_evaluate.py --compare mobilenet_v2__full:mobilenet_v2__baseline
python scripts/05_figure3.py
```
Đổi tham số bất kỳ không cần sửa file: `--set lr=3e-4 batch_size=16`.

Kết quả:
- `runs/<backbone>__<variant>/` — model từng fold, dự đoán CV (`cv_predictions.csv`) và test (`test_predictions.csv`)
- `results/cv_summary.csv` — mean ± SD qua 5 fold
- `results/test_summary.csv` — tập test với 95% CI (bootstrap 1.000 lần theo bệnh nhân)
- `results/comparisons.csv` — paired bootstrap, DeLong, McNemar, NRI, chênh lệch theo nhóm Fitzpatrick
- `results/test_with_adtss__*.csv` — AD-TSS từng ảnh

## Những điểm bài không ghi rõ — code đã chọn như sau

| Điểm | Lựa chọn trong code | Đổi ở đâu |
|---|---|---|
| Early stopping dựa trên tập nào | Tách 15% bệnh nhân **bên trong** fold huấn luyện (inner-val) để early stopping + temperature scaling; fold validation ngoài không bị đụng tới | `inner_val_frac` (đặt 0 = dùng luôn fold validation) |
| Dự đoán trên tập test | Trung bình xác suất của 5 model fold | `test_mode: refit` = huấn luyện lại trên toàn bộ dev |
| DRW "sau epoch 20" | Đếm epoch tổng (kể cả 10 epoch đóng băng) | `drw_start_epoch` |
| Vừa sampler cân bằng vừa DRW | Bật cả hai như bài viết | `balanced_sampler`, `drw` |
| Định nghĩa NRI | NRI phân loại đa lớp: với mỗi lớp thật, (tỉ lệ sai→đúng) − (tỉ lệ đúng→sai), lấy trung bình 3 lớp | `metrics.nri_multiclass` |
| Ngưỡng AD-TSS | Làm tròn 2 chữ số rồi chia ≤0,33 / ≤0,66 / còn lại | `adtss.py` |
| Hình 3D: "predicted phenotype" lấy từ đâu | Thêm nhánh phụ dự đoán hình thái (`aux_phenotype: true`); bin dự đoán = mức độ dự đoán + hình thái có xác suất cao hơn trong 2 hình thái của mức đó | `aux_phenotype`, `phenotype_bins` |
| Ảnh tham chiếu tông da cho style transfer | Nên dùng thư mục mảng da thường của bệnh nhân IV–VI (`--style-dir`); nếu không có, lấy ảnh IV–VI trong tập dev | `02_style_augment.py` |
| Fitzpatrick trong metadata fusion | Có đưa vào (như mục 2.9) | `meta_categorical` |

## Chạy thử nhanh không cần dữ liệu thật

```bash
python tests/make_dummy_data.py dummy 40
python scripts/01_preprocess.py --metadata dummy/metadata_raw.csv --raw-root dummy/raw --out-root dummy/pre --out-csv dummy/metadata.csv
python scripts/03_run_cv.py --backbone mobilenet_v2 --variant full \
  --set metadata_csv=dummy/metadata.csv image_root=dummy/pre pretrained=false freeze_epochs=1 finetune_epochs=1 num_workers=0
```
Ảnh giả chỉ để kiểm tra code chạy, số liệu ra không có ý nghĩa.
