import json
from typing import Optional, Tuple
import numpy as np
import glob
import os
import random
from tqdm import tqdm
from penn_utils import NUM_JOINTS_PENN
from ubnormal_loader import coco17_to_penn13_flat_xy

def load_json_pose(
    json_path: str,
    min_len: int = 1,
    target_len: Optional[int] = None,  # <-- added
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load an AlphaPose JSON and return a (T, 26) sequence in PennAction format.
    Optionally uniformly resample to `target_len` frames.
    """
    fname = os.path.basename(json_path)
    label = 1 if fname.startswith("abnormal") else 0  # ShanghaiTech will just always give 0

    with open(json_path, "r") as f:
        data = json.load(f)

    best_seq = None
    best_T = 0

    for _, frames in data.items():
        frame_keys = sorted(frames.keys(), key=lambda x: int(x))
        T = len(frame_keys)
        if T < min_len:
            continue

        poses = np.zeros((T, NUM_JOINTS_PENN * 2), dtype=np.float32)

        for i, fr_key in enumerate(frame_keys):
            kps = frames[fr_key]["keypoints"]
            poses[i] = coco17_to_penn13_flat_xy(kps)

        if T > best_T:
            best_T = T
            best_seq = poses

    if best_seq is None:
        return None, None

    # -------------------------------
    # 🔥 Uniform resampling happens here
    # -------------------------------
    if target_len is not None and best_T > 0:
        idx = np.linspace(0, best_T - 1, num=target_len).astype(int)
        best_seq = best_seq[idx]  # (target_len, 26)

    return best_seq.astype(np.float32), int(label)


def build_shanghaitech_lists(root: str):
    """
    For masked pretraining: collect ALL clips from ShanghaiTech/pose/{train,test}
    and convert COCO17 -> PennAction13 (26 dims per frame).
    """
    pose_root = os.path.join(root, "pose")

    all_seq = []
    all_lbl = []   # optional: ShanghaiTech has no normal/abnormal label
    target_len = 512 # uniform resample length

    for split in ["train", "test"]:
        split_dir = os.path.join(pose_root, split)
        json_files = sorted(
            glob.glob(os.path.join(split_dir, "*_alphapose_tracked_person.json"))
        )

        for jp in tqdm(json_files, desc=f"Collect {split}"):
            seq, _ = load_json_pose(jp, target_len=target_len)   # label ignored
            if seq is None:
                continue
            all_seq.append(seq)
            all_lbl.append(0)   # dummy label

    # Shuffle
    combined = list(zip(all_seq, all_lbl))
    random.shuffle(combined)
    all_seq[:], all_lbl[:] = zip(*combined)

    print(f"[ShanghaiTech] total clips={len(all_seq)}")
    return list(all_seq), list(all_lbl)


