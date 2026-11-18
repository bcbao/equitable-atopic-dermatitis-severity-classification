#!/usr/bin/env python
"""Section 2.8: train one LoRA adapter per cross-validation split.

Each adapter is trained on Stable Diffusion v1.5 using only the
Fitzpatrick IV–VI images of that split's training subset, so no validation
or held-out test image is ever seen by the generator.

  python scripts/02a_train_lora.py --metadata data/metadata.csv \
      --image-root data/preprocessed --out . --folds 1,2,3,4,5

Adapters are written to ./lora_fold_1 ... ./lora_fold_5.

Defaults reproduce the manuscript: rank 16, alpha 16, 1,200 steps,
learning rate 1e-4, batch size 2, 512 × 512 pixels, mixed precision.
Requires a CUDA GPU (the study used an NVIDIA RTX A6000, 48 GB).
Tested with diffusers 0.25.0, peft 0.7, transformers 4.36, PyTorch 2.1.0.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from adpipe.data import load_metadata, make_patient_folds
from adpipe.generative import BASE_MODEL, lora_caption, lora_dir, split_training_frames


class LoraImages(Dataset):
    def __init__(self, df, root, tokenizer, size):
        self.paths = [os.path.join(root, p) for p in df.image_path]
        self.captions = [lora_caption(r) for _, r in df.iterrows()]
        self.tok = tokenizer
        self.tf = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BILINEAR), T.RandomCrop(size),
                             T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        ids = self.tok(self.captions[i], padding="max_length", truncation=True,
                       max_length=self.tok.model_max_length, return_tensors="pt").input_ids[0]
        return {"pixel_values": self.tf(Image.open(self.paths[i]).convert("RGB")), "input_ids": ids}


def train_one(images, out_dir, a, device):
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(a.seed)
    tok = CLIPTokenizer.from_pretrained(a.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(a.base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(a.base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(a.base_model, subfolder="unet")
    sched = DDPMScheduler.from_pretrained(a.base_model, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    weight_dtype = torch.float16 if a.mixed_precision == "fp16" else torch.float32
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device)  # trainable LoRA weights stay in float32
    unet.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.alpha, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2)
    dl = DataLoader(LoraImages(images, a.image_root, tok, a.resolution), batch_size=a.batch_size,
                    shuffle=True, num_workers=a.num_workers, drop_last=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(weight_dtype == torch.float16))
    step = 0
    unet.train()
    log = []
    while step < a.steps:
        for b in dl:
            with torch.no_grad():
                x = b["pixel_values"].to(device, dtype=weight_dtype)
                latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                enc = text_encoder(b["input_ids"].to(device))[0]
            noise = torch.randn_like(latents)
            t = torch.randint(0, sched.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy = sched.add_noise(latents, noise, t)
            target = noise if sched.config.prediction_type == "epsilon" else sched.get_velocity(latents, noise, t)
            with torch.autocast("cuda", dtype=weight_dtype, enabled=(weight_dtype == torch.float16)):
                pred = unet(noisy, t, enc).sample
            loss = F.mse_loss(pred.float(), target.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 100 == 0:
                print(f"    step {step}/{a.steps}  loss {loss.item():.4f}")
                log.append({"step": step, "loss": loss.item()})
            if step >= a.steps:
                break
    os.makedirs(out_dir, exist_ok=True)
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(save_directory=out_dir, unet_lora_layers=lora, safe_serialization=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/metadata.csv")
    ap.add_argument("--image-root", default="data/preprocessed")
    ap.add_argument("--out", default=".", help="adapters go to <out>/lora_fold_<k>")
    ap.add_argument("--folds", default="1,2,3,4,5", help="fold numbers 1–5")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=42, help="must match the cross-validation seed")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--mixed-precision", choices=["fp16", "no"], default="fp16")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="only list the images per fold")
    a = ap.parse_args()

    df = load_metadata(a.metadata)
    dev = df[df.split == "dev"].reset_index(drop=True)
    folds = make_patient_folds(dev, a.n_folds, a.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = []
    for k in [int(f) for f in a.folds.split(",")]:
        train = split_training_frames(dev, folds, k - 1)
        iv_vi = train[train.fitz_group == "IV-VI"].reset_index(drop=True)
        print(f"fold {k}: {len(iv_vi)} Fitzpatrick IV–VI training images for LoRA")
        out_dir = lora_dir(a.out, k)
        os.makedirs(out_dir, exist_ok=True)
        iv_vi.assign(caption_used=[lora_caption(r) for _, r in iv_vi.iterrows()]).to_csv(
            os.path.join(out_dir, "lora_training_images.csv"), index=False)
        summary.append({"fold": k, "n_iv_vi_images": len(iv_vi)})
        if a.dry_run:
            continue
        if device.type != "cuda":
            sys.exit("LoRA training needs a CUDA GPU")
        log = train_one(iv_vi, out_dir, a, device)
        json.dump({"args": vars(a), "fold": k, "n_images": len(iv_vi), "loss_log": log},
                  open(os.path.join(out_dir, "train_log.json"), "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
