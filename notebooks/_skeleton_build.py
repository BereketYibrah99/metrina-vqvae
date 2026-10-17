"""
Scratch script to derive the final PARENTS array for src/skeleton.py.
Run once; output is hand-verified then hardcoded into src/skeleton.py.
Not part of the reusable src/ package.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.manifest import load_manifest, filter_52_joint_min_length

RAW_DIR = Path(__file__).resolve().parents[1] / "data/raw/motion_text_7000"
J = 52


def load_all(ids):
    for i in ids:
        yield np.load(RAW_DIR / "joints" / f"{i}.npy").astype(np.float64)


def pairwise_dist_stats(ids):
    sum_sq = np.zeros((J, J))
    sum_ = np.zeros((J, J))
    n = 0
    for a in load_all(ids):
        diff = a[:, :, None, :] - a[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        sum_sq += (dist**2).sum(0)
        sum_ += dist.sum(0)
        n += a.shape[0]
    mean = sum_ / n
    var = sum_sq / n - mean**2
    return mean, var


def prim_mst(var, mean, root, candidates):
    """MST restricted to `candidates` (list of joint indices), rooted at `root`."""
    INF = 1e18
    idx = candidates
    n = len(idx)
    pos = {j: k for k, j in enumerate(idx)}
    w = np.full((n, n), INF)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            ja, jb = idx[a], idx[b]
            if mean[ja, jb] < 1e-6:
                continue
            w[a, b] = var[ja, jb]
    visited = [False] * n
    parent = [-1] * n
    key = [INF] * n
    key[pos[root]] = 0
    for _ in range(n):
        u = min((k for k in range(n) if not visited[k]), key=lambda k: key[k])
        visited[u] = True
        for v in range(n):
            if not visited[v] and w[u][v] < key[v]:
                key[v] = w[u][v]
                parent[v] = u
    result = {}
    for k in range(n):
        result[idx[k]] = idx[parent[k]] if parent[k] >= 0 else -1
    return result


def main():
    manifest = load_manifest(RAW_DIR / "manifest.csv")
    ids = filter_52_joint_min_length(manifest, RAW_DIR, min_frames=24)
    print(f"using {len(ids)} sequences for skeleton derivation")

    mean, var = pairwise_dist_stats(ids)
    np.save("/tmp/skel_mean.npy", mean)
    np.save("/tmp/skel_var.npy", var)

    # --- Step 1: body tree (joints 0-21), isolated submatrix ---
    body = list(range(22))
    body_parents = prim_mst(var, mean, root=0, candidates=body)
    print("\n=== body tree (0-21) ===")
    for j in range(22):
        print(j, "<-", body_parents[j], f"len={mean[j][body_parents[j]]:.4f}" if body_parents[j] >= 0 else "")

    # --- Step 1b: left/right disambiguation, root-relative x-sign ---
    print("\n=== left/right x-sign check (root-relative) ===")
    sums = {j: 0.0 for j in range(22)}
    counts = {j: 0 for j in range(22)}
    for a in load_all(ids):
        root_x = a[:, 0, 0:1]
        rel_x = a[:, :, 0] - root_x  # (T, J)
        for j in range(22):
            sums[j] += rel_x[:, j].sum()
            counts[j] += rel_x.shape[0]
    for j in range(22):
        print(j, "mean_rel_x=", sums[j] / counts[j])

    # --- Step 2: hand assignment by nearest-wrist distance ---
    # body_parents gives us wrists; identify them as the joints among 18-21 with
    # max chain depth on each side, but simplest: wrists are whichever of
    # {18,19,20,21} are leaves of the body tree (no children in body_parents)
    children_count = {j: 0 for j in body}
    for j, p in body_parents.items():
        if p >= 0:
            children_count[p] = children_count.get(p, 0) + 1
    leaves = [j for j in body if children_count.get(j, 0) == 0]
    print("\nbody leaves (candidates for wrists/head/feet):", leaves)

    fingers = list(range(22, 52))
    # assume wrists are the two leaves with the most nearby "finger" joints
    wrist_candidates = leaves
    dist_to_wrist = {}
    for wc in wrist_candidates:
        d = mean[wc, fingers].mean()
        dist_to_wrist[wc] = d
    print("mean dist body-leaf -> all finger joints:", dist_to_wrist)

    best_wrists = sorted(wrist_candidates, key=lambda w: dist_to_wrist[w])[:2]
    print("chosen wrists:", best_wrists)

    hand_assign = {}
    for f in fingers:
        d0 = mean[f, best_wrists[0]]
        d1 = mean[f, best_wrists[1]]
        hand_assign[f] = best_wrists[0] if d0 < d1 else best_wrists[1]

    for w in best_wrists:
        members = [f for f in fingers if hand_assign[f] == w]
        print(f"hand rooted at wrist {w}: {len(members)} joints -> {members}")

    # --- Step 3: per-hand depth clustering + constrained chain linking ---
    finger_parents = {}
    for w in best_wrists:
        members = [f for f in fingers if hand_assign[f] == w]
        dist_from_wrist = {f: mean[f, w] for f in members}
        ordered = sorted(members, key=lambda f: dist_from_wrist[f])
        # 3 clusters of 5 by rank order (proximal, middle, distal)
        proximal = ordered[0:5]
        middle = ordered[5:10]
        distal = ordered[10:15]
        for f in proximal:
            finger_parents[f] = w
        for f in proximal:
            # nearest middle-cluster joint by variance
            best = min(middle, key=lambda m: var[f, m])
            finger_parents[best] = f
        # for the assigned middle joints, link to nearest distal by variance
        assigned_middle = [finger_parents[f] for f in proximal]
        for m in assigned_middle:
            best = min(distal, key=lambda d: var[m, d])
            finger_parents[best] = m

    print("\n=== finger parents ===")
    for f in sorted(finger_parents):
        print(f, "<-", finger_parents[f], f"len={mean[f][finger_parents[f]]:.4f}")

    # --- assemble full PARENTS array ---
    parents = [-1] * 52
    for j in range(1, 22):
        parents[j] = body_parents[j]
    for f, p in finger_parents.items():
        parents[f] = p

    print("\n=== FULL PARENTS ===")
    print(parents)

    # sanity: every joint 1..51 assigned
    missing = [j for j in range(1, 52) if parents[j] == -1]
    print("missing assignments:", missing)


if __name__ == "__main__":
    main()
