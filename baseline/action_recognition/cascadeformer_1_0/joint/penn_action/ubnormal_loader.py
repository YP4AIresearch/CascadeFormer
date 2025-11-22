import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import glob
import json
from typing import List, Tuple, Optional
from tqdm import tqdm

NUM_JOINTS_PENN = 13      # Penn Action-style keypoints
NUM_JOINTS_UBNORMAL = 17  # COCO-style keypoints

def coco17_to_penn13_flat_xy(kps: list) -> np.ndarray:
    """
    kps: flat list of length 51 = 17 * (x, y, score)
    Return: (26,) = 13 * (x, y) in Penn Action order.
    """
    arr = np.array(kps, dtype=np.float32).reshape(-1, 3)  # (17,3)
    xy = arr[:, :2]                                       # (17,2)

    penn = np.zeros((NUM_JOINTS_PENN, 2), dtype=np.float32)

    # 1) Head = average of nose + eyes + ears (0..4)
    penn[0] = xy[0:5].mean(axis=0)

    # 2) Upper body (same semantics as we drew)
    penn[1] = xy[5]   # L shoulder
    penn[2] = xy[6]   # R shoulder
    penn[3] = xy[7]   # L elbow
    penn[4] = xy[8]   # R elbow
    penn[5] = xy[9]   # L wrist
    penn[6] = xy[10]  # R wrist

    # 3) Lower body
    penn[7]  = xy[11] # L hip
    penn[8]  = xy[12] # R hip
    penn[9]  = xy[13] # L knee
    penn[10] = xy[14] # R knee
    penn[11] = xy[15] # L ankle
    penn[12] = xy[16] # R ankle

    return penn.reshape(-1)  # (26,)



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def keypoints_to_flat_xy(kps: list) -> np.ndarray:
    """
    UBnormal AlphaPose format:
      kps: flat list of length 51 = 17 * (x, y, score)
    Return: (34,) = 17 * (x, y)
    """
    arr = np.array(kps, dtype=np.float32).reshape(-1, 3)  # (17,3)
    xy = arr[:, :2]                                       # (17,2)
    return xy.reshape(-1)                                 # (34,)


def load_json_pose(
    json_path: str,
    min_len: int = 1,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load a UBnormal AlphaPose JSON and turn it into a (T, 34) sequence.
    """
    fname = os.path.basename(json_path)
    label = 1 if fname.startswith("abnormal") else 0

    with open(json_path, "r") as f:
        data = json.load(f)   # track_id(str) -> frame_str -> {keypoints, scores}

    best_seq = None     # (T, 34) before resampling
    best_T = 0          # original length

    for _, frames in data.items():
        # keep original keys, sorted numerically
        frame_keys = sorted(frames.keys(), key=lambda x: int(x))
        T = len(frame_keys)
        if T < min_len:
            continue
 
        # NOTE: originally we had 17 joints for UBnormal, but we adapt to 13-joint Penn Action format
        # poses = np.zeros((T, NUM_JOINTS_UBNORMAL * 2), dtype=np.float32)  # (T,34)
        poses = np.zeros((T, NUM_JOINTS_PENN * 2), dtype=np.float32)  # (T,26)

        for i, fr_key in enumerate(frame_keys):
            kps = frames[fr_key]["keypoints"]
            # poses[i] = keypoints_to_flat_xy(kps)
            poses[i] = coco17_to_penn13_flat_xy(kps)            

        if T > best_T:
            best_T = T
            best_seq = poses

    if best_seq is None:
        return None, None

    return best_seq.astype(np.float32), int(label)


def _collect_split(root_pose_dir: str, split: str
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Collect sequences + labels from a given split under UBnormal/pose/*

    split in {"train", "abnormal_train", "test"}
    """
    pose_dir = os.path.join(root_pose_dir, split)
    json_files = sorted(
        glob.glob(os.path.join(pose_dir, "*_alphapose_tracked_person.json"))
    )

    seqs: List[np.ndarray] = []
    lbls: List[int] = []

    for jp in tqdm(json_files, desc=f"Collecting {split}"):
        seq, lbl = load_json_pose(jp)
        if seq is None:
            continue
        seqs.append(seq)
        lbls.append(lbl)

    print(f"[{split}] clips={len(seqs)} "
          f"| abnormal={sum(lbls)} | normal={len(lbls)-sum(lbls)}")
    return seqs, lbls


def build_ubnormal_lists(root: str
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    For masked pretraining: merge ALL splits into one dataset.

    Returns:
        all_seq, all_lbl
    where each seq is (T, 26) (Penn-mapped) and label in {0,1}.
    Labels are kept only for optional stratified val split or analysis.
    """
    pose_root = os.path.join(root, "pose")

    all_seq: List[np.ndarray] = []
    all_lbl: List[int] = []

    for split in ["train", "abnormal_train", "test"]:
        split_seq, split_lbl = _collect_split(pose_root, split)
        all_seq.extend(split_seq)
        all_lbl.extend(split_lbl)

    # shuffle everything
    combined = list(zip(all_seq, all_lbl))
    random.shuffle(combined)
    all_seq[:], all_lbl[:] = zip(*combined)

    print(f"#all clips={len(all_seq)} | abnormal={sum(all_lbl)} | normal={len(all_lbl)-sum(all_lbl)}")
    return list(all_seq), list(all_lbl)


def split_train_val(train_seq, train_lbl, val_ratio=0.15, seed=42):
    tr_idx, val_idx = train_test_split(
        np.arange(len(train_seq)),
        test_size=val_ratio,
        random_state=seed,
        stratify=train_lbl
    )
    tr_seq  = [train_seq[i] for i in tr_idx]
    tr_lbl  = [train_lbl[i] for i in tr_idx]
    val_seq = [train_seq[i] for i in val_idx]
    val_lbl = [train_lbl[i] for i in val_idx]

    return tr_seq, tr_lbl, val_seq, val_lbl

