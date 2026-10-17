import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import MotionDataset
from src.data.features import feature_to_joints
from src.data.normalize import NormStats
from src.data.splits import load_splits
from src.model.vqvae import VQVAE
from src.train import get_device, train
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
raw_dir = ROOT / "data/raw/motion_text_7000"
splits = load_splits(ROOT / "data/splits")
stats_j = NormStats.load(ROOT / "data/stats/stats_joints.npz")

device = get_device()
print("device:", device)

# --- Check 3: untrained-model sanity ---
print("\n=== Check 3: untrained-model MPJPE sanity ===")
ds = MotionDataset(splits["test"][:20], raw_dir, "joints", stats_j, crop_len=64)
loader = DataLoader(ds, batch_size=20)
batch = next(iter(loader)).to(device)

model = VQVAE(feature_dim=156, code_dim=512, num_codes=512).to(device)
model.eval()
with torch.no_grad():
    out = model(batch)
    recon_norm = out["recon"].cpu().numpy()
    gt_norm = batch.cpu().numpy()
    recon_joints = stats_j.denormalize(recon_norm).reshape(-1, 64, 52, 3)
    gt_joints = stats_j.denormalize(gt_norm).reshape(-1, 64, 52, 3)
    mpjpe_mm = np.linalg.norm(recon_joints - gt_joints, axis=-1).mean() * 1000
    print(f"untrained MPJPE: {mpjpe_mm:.1f} mm (should be large / non-trivial, not ~0)")

    usage = out["usage"].cpu().numpy()
    active = (usage > 0).sum()
    probs = usage / usage.sum()
    probs = probs[probs > 0]
    perplexity = np.exp(-(probs * np.log(probs)).sum())
    print(f"untrained codebook: active={active}/512, perplexity={perplexity:.1f} "
          f"(should be near-max entropy at init, not collapsed to 1-2 codes)")

# --- Check 4: overfit a tiny subset ---
print("\n=== Check 4: overfit tiny subset (15 sequences, no val split) ===")
tiny_ids = splits["train"][:15]
ds_tiny = MotionDataset(tiny_ids, raw_dir, "joints", stats_j, crop_len=64)
print(f"tiny dataset size (after crop-length filter): {len(ds_tiny)}")
loader_tiny = DataLoader(ds_tiny, batch_size=len(ds_tiny), shuffle=True)

model_tiny = VQVAE(feature_dim=156, code_dim=512, num_codes=512)
history = train(
    model_tiny, loader_tiny, loader_tiny, device,
    epochs=300, lr=3e-4, log_every=50,
)
print(f"\nfinal train_l1: {history[-1]['train_l1']:.6f} (should be near 0 if model/training loop is correct)")
