import os
import numpy as np
import glob
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from base_dataset import ActionRecognitionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

NUM_JOINTS_NTU = 25
OFFICIAL_XSUB_TRAIN_SUBJECTS = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
OFFICIAL_XVIEW_TRAIN_CAMERAS = [2, 3]

def normalize_scale(skeleton: np.ndarray) -> np.ndarray:
    """
    Normalize the skeleton so that the scale is invariant.
    E.g., divide by std or max joint distance to origin.
    """
    scale = np.linalg.norm(skeleton, axis=-1).max() + 1e-8
    return skeleton / scale

def random_rotation(sequence: np.ndarray, max_angle=15) -> np.ndarray:
    angle = np.deg2rad(np.random.uniform(-max_angle, max_angle))
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos, -sin, 0],
        [sin,  cos, 0],
        [0,     0,  1]
    ])
    return np.einsum('tjd,dk->tjk', sequence, R)

def random_scaling(sequence: np.ndarray, scale_range=(0.8, 1.2)) -> np.ndarray:
    scale = np.random.uniform(*scale_range)
    return sequence * scale

def add_gaussian_noise(sequence: np.ndarray, std=0.01) -> np.ndarray:
    noise = np.random.normal(0, std, sequence.shape)
    return sequence + noise

def temporal_jitter(sequence: np.ndarray, max_jitter=5) -> np.ndarray:
    T = sequence.shape[0]
    jitter = np.random.randint(-max_jitter, max_jitter + 1, size=T)
    jitter = np.clip(np.arange(T) + jitter, 0, T - 1)
    return sequence[jitter]

def temporal_crop(sequence: np.ndarray, crop_len=64) -> np.ndarray:
    T = sequence.shape[0]
    if T <= crop_len:
        return sequence
    start = np.random.randint(0, T - crop_len)
    return sequence[start:start + crop_len]

def transform_sequence(sequence):
    sequence = random_rotation(sequence)
    sequence = random_scaling(sequence)
    sequence = add_gaussian_noise(sequence)
    if np.random.rand() < 0.5:
        sequence = temporal_jitter(sequence)
    if np.random.rand() < 0.5:
        sequence = temporal_crop(sequence)
    return normalize_scale(sequence).astype(np.float32)

def read_ntu_skeleton_file(filepath: str, num_joints: int = NUM_JOINTS_NTU, T_out: int = 64, p: float = 1.0) -> np.ndarray:
    """
    Read an NTU RGB+D .skeleton file and apply trimmed-uniform random sampling
    to get T_out frames (shape: [T_out, num_joints, 3]).

    Parameters:
        - filepath: path to the .skeleton file
        - num_joints: number of joints (default = 25)
        - T_out: number of frames to sample (e.g., 64)
        - p: portion of total sequence to consider (e.g., 0.9 trims 5% on each end)

    Returns:
        - xyz_sampled: np.ndarray of shape (T_out, num_joints, 3)
    """
    with open(filepath, "r") as f:
        total_frames = int(f.readline().strip())
        xyz = np.zeros((total_frames, num_joints, 3), dtype=np.float32)

        for t in range(total_frames):
            body_cnt = int(f.readline().strip())

            if body_cnt > 0:
                _ = f.readline()  # body ID info
                _ = f.readline()  # body metadata

                for j in range(num_joints):
                    vals = list(map(float, f.readline().strip().split()))
                    xyz[t, j] = vals[:3]

                # Skip other bodies
                for _ in range(body_cnt - 1):
                    _ = f.readline()
                    _ = f.readline()
                    for _ in range(num_joints):
                        _ = f.readline()
    
    """
        trimmed-uniform random sampling adapted from SkateFormer (https://arxiv.org/abs/2403.09508)
    """
    # Trim beginning and end based on p
    valid_len = int(total_frames * p)
    if valid_len < T_out:
        # Repeat to make sure we can sample T_out frames
        repeats = (T_out + valid_len - 1) // valid_len
        xyz = np.tile(xyz, (repeats + 1, 1, 1))
        total_frames = xyz.shape[0]
        valid_len = int(total_frames * p)

    start = int((total_frames - valid_len) / 2)
    end = start + valid_len

    trimmed_xyz = xyz[start:end]  # Shape: [valid_len, V, 3]

    # Divide into T_out intervals and sample one frame from each
    interval_len = valid_len / T_out
    sampled = []
    for i in range(T_out):
        interval_start = int(i * interval_len)
        interval_end = int((i + 1) * interval_len)
        if interval_end > valid_len:
            interval_end = valid_len
        idx = random.randint(interval_start, max(interval_start, interval_end - 1))
        sampled.append(trimmed_xyz[idx])

    xyz_sampled = np.stack(sampled, axis=0)  # (T_out, V, 3)
    return xyz_sampled.astype(np.float32)


def build_ntu_skeleton_lists_xsub(
    skeleton_root: str,
    is_train: bool = True,
    train_subjects: List[int] = OFFICIAL_XSUB_TRAIN_SUBJECTS,
    num_joints: int = NUM_JOINTS_NTU,
    augment: bool = True
) -> Tuple[List[np.ndarray], List[int]]:
    sequences, labels = [], []

    for filepath in tqdm(sorted(glob.glob(os.path.join(skeleton_root, '*.skeleton')))):
        filename = os.path.basename(filepath)
        subject_id = int(filename[9:12])

        if (is_train and subject_id not in train_subjects) or (not is_train and subject_id in train_subjects):
            continue

        action_idx = int(filename[17:20]) - 1


        """
            Here, start reading data + apply augmentations
        """
        skeleton = read_ntu_skeleton_file(filepath, num_joints)
        skeleton = skeleton - skeleton[:, 0:1, :] # hip centering

        # always apply normalization
        skeleton = normalize_scale(skeleton)
        skeleton = skeleton.astype(np.float32)

        # HAHA! LET'S DO SOME AUGMENTATIONS
        if augment and is_train:
            skeleton = transform_sequence(skeleton)

        sequences.append(skeleton)
        labels.append(action_idx)

    # cache them
    if is_train and augment:
        save_cached_data(sequences, labels, path="ntu_cache_train_sub_64_10_augmented.npz")
    elif is_train and not augment:
        save_cached_data(sequences, labels, path="ntu_cache_train_sub_64_10_validation.npz")
    else:
        save_cached_data(sequences, labels, path="ntu_cache_test_sub_64_10.npz")

    return sequences, labels

def build_ntu_skeleton_lists_xview(
    skeleton_root: str,
    is_train: bool = True,
    train_cameras: List[int] = OFFICIAL_XVIEW_TRAIN_CAMERAS,
    num_joints: int = NUM_JOINTS_NTU,
    augment: bool = True
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Build NTU RGB+D dataset using Cross-View split.
    Args:
        skeleton_root: path to folder with .skeleton files
        is_train: True for train split, False for test split
        train_cameras: list of camera IDs used for training (default: [2, 3])
        num_joints: number of joints (default: 25 for NTU)
    Returns:
        sequences, labels
    """
    sequences, labels = [], []

    for filepath in tqdm(sorted(glob.glob(os.path.join(skeleton_root, '*.skeleton')))):
        filename = os.path.basename(filepath)
        camera_id = int(filename[5:8])  # extract "C###" → camera ID

        # Check if sample belongs to current split
        if (is_train and camera_id not in train_cameras) or (not is_train and camera_id in train_cameras):
            continue

        # Extract action label (zero-based)
        action_idx = int(filename[17:20]) - 1

        """
        Read skeleton data and apply augmentations
        """
        skeleton = read_ntu_skeleton_file(filepath, num_joints)
        skeleton = skeleton - skeleton[:, 0:1, :]  # hip centering
        # always apply normalization
        skeleton = normalize_scale(skeleton)
        skeleton = skeleton.astype(np.float32)
        # Apply augmentations only for training set
        if is_train:
            skeleton = transform_sequence(skeleton)
        sequences.append(skeleton)
        labels.append(action_idx)
    # cache them
    if is_train and augment:
        save_cached_data(sequences, labels, path="ntu_cache_train_view_64_10_augmented.npz")
    elif is_train and not augment:
        save_cached_data(sequences, labels, path="ntu_cache_train_view_64_10_validation.npz")
    else:
        save_cached_data(sequences, labels, path="ntu_cache_test_view_64_10.npz")
    return sequences, labels


def split_train_val(
    sequences: List[np.ndarray],
    labels: List[int],
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[np.ndarray], List[int], List[np.ndarray], List[int]]:
    """
    Splits the NTU dataset into train and validation sets.
    """
    indices = np.arange(len(sequences))
    tr_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels
    )

    tr_seq  = [sequences[i] for i in tr_idx]
    tr_lbl  = [labels[i] for i in tr_idx]
    val_seq = [sequences[i] for i in val_idx]
    val_lbl = [labels[i] for i in val_idx]

    return tr_seq, tr_lbl, val_seq, val_lbl

def save_cached_data(sequences, labels, path="ntu_cache_train_sub.npz"):
    # Make sure all sequences are the same shape
    for i, seq in enumerate(sequences):
        assert seq.shape == sequences[0].shape, f"Shape mismatch at {i}: {seq.shape}"
        assert seq.dtype == np.float32, f"Dtype mismatch at {i}: {seq.dtype}"

    sequences_arr = np.stack(sequences, axis=0).astype(np.float32)  # ✅ consistent array
    labels_arr = np.array(labels, dtype=np.int64)
    np.savez_compressed(path, sequences=sequences_arr, labels=labels_arr)


if __name__ == "__main__":
    import time
    t_start = time.time()
    train_seq, train_lbl = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=True, augment=True)
    all_seq_clean, all_lbl_clean = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=True, augment=False)
    _, _, val_seq, val_lbl = split_train_val(all_seq_clean, all_lbl_clean, val_ratio=0.15)
    test_seq, test_lbl = build_ntu_skeleton_lists_xsub('nturgb+d_skeletons', is_train=False, augment=False)
    t_end = time.time()

    print(f"Time taken: {t_end - t_start:.2f} seconds")
    print(f"Train sequences: {len(train_seq)}, Train labels: {len(train_lbl)}")
    print(f"Validation sequences: {len(val_seq)}, Validation labels: {len(val_lbl)}")
    print(f"Sample train sequence shape: {train_seq[0].shape}, dtype: {train_seq[0].dtype}")
    print(f"Sample validation sequence shape: {val_seq[0].shape}, dtype: {val_seq[0].dtype}")  

    train_dataset = ActionRecognitionDataset(train_seq, train_lbl)
    val_dataset = ActionRecognitionDataset(val_seq, val_lbl)
    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Validation DataLoader: {len(val_loader)} batches")
    print(f"Test DataLoader: {len(test_loader)} batches")
