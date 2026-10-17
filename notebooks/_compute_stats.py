import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import to_bones_feature, to_joints_feature
from src.data.normalize import compute_stats
from src.data.splits import load_splits

ROOT = Path(__file__).resolve().parents[1]
raw_dir = ROOT / "data/raw/motion_text_7000"
splits = load_splits(ROOT / "data/splits")
train_ids = splits["train"]
print(f"computing stats over {len(train_ids)} training sequences")


def joints_iter(ids):
    for i in ids:
        j = np.load(raw_dir / "joints" / f"{i}.npy").astype(np.float32)
        yield to_joints_feature(j)


def bones_iter(ids):
    for i in ids:
        j = np.load(raw_dir / "joints" / f"{i}.npy").astype(np.float32)
        yield to_bones_feature(j)


stats_dir = ROOT / "data/stats"
stats_dir.mkdir(parents=True, exist_ok=True)

stats_joints = compute_stats(joints_iter(train_ids))
stats_joints.save(stats_dir / "stats_joints.npz")
print("joints stats: mean range", stats_joints.mean.min(), stats_joints.mean.max(),
      "std range", stats_joints.std.min(), stats_joints.std.max())

stats_bones = compute_stats(bones_iter(train_ids))
stats_bones.save(stats_dir / "stats_bones.npz")
print("bones stats: mean range", stats_bones.mean.min(), stats_bones.mean.max(),
      "std range", stats_bones.std.min(), stats_bones.std.max())


def bones_body_iter(ids):
    for i in ids:
        j = np.load(raw_dir / "joints" / f"{i}.npy").astype(np.float32)
        yield to_bones_feature(j, body_only=True)


stats_bones_body = compute_stats(bones_body_iter(train_ids))
stats_bones_body.save(stats_dir / "stats_bones_body.npz")
print("bones (body-only) stats: mean range", stats_bones_body.mean.min(), stats_bones_body.mean.max(),
      "std range", stats_bones_body.std.min(), stats_bones_body.std.max())


def bones_only_body_iter(ids):
    for i in ids:
        j = np.load(raw_dir / "joints" / f"{i}.npy").astype(np.float32)
        yield to_bones_feature(j, body_only=True, include_translation=False)


stats_bones_only_body = compute_stats(bones_only_body_iter(train_ids))
stats_bones_only_body.save(stats_dir / "stats_bones_only_body.npz")
print("bones-only (no translation, body-only) stats: mean range",
      stats_bones_only_body.mean.min(), stats_bones_only_body.mean.max(),
      "std range", stats_bones_only_body.std.min(), stats_bones_only_body.std.max())
