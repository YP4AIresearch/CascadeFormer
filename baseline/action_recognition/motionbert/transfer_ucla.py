import torch
import numpy as np
from typing import Tuple
from tqdm import tqdm
import torch.nn.functional as F
import random
import argparse
from torch.utils.data import DataLoader
from base_dataset import ActionRecognitionDataset
from SF_UCLA_loader import SF_UCLA_Dataset, skateformer_collate_fn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="NTU -> NW-UCLA 1-shot eval (training-free) with CascadeFormer")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_runs", type=int, default=10)
    return parser.parse_args()



def ucla20_to_ntu25(skel_batch: torch.Tensor) -> torch.Tensor:
    """
    skel_batch:
        (B, T, 60)  or  (B, T, 20, 3)
        representing UCLA 20-joint skeletons (x,y,z per joint).
    returns:
        (B, T, 25, 3) in NTU 25-joint layout.
    """
    if skel_batch.dim() == 3:
        B, T, F = skel_batch.shape
        assert F == 60, f"Expected 60 features (20*3), got {F}"
        skel_batch = skel_batch.view(B, T, 20, 3)
    elif skel_batch.dim() == 4:
        B, T, J, C = skel_batch.shape
        assert J == 20 and C == 3, f"Expected (20,3), got ({J},{C})"
    else:
        raise ValueError(f"Unexpected skel_batch shape {skel_batch.shape}")

    B, T, J, C = skel_batch.shape
    device = skel_batch.device
    dtype  = skel_batch.dtype

    u = skel_batch # (B, T, 20, 3)
    out = torch.zeros(B, T, 25, 3, device=device, dtype=dtype)
    
    # mapping: torso
    out[:, :, 0, :] = u[:, :, 0, :]   # N1  <- U1  hipCenter
    out[:, :, 1, :] = u[:, :, 1, :]   # N2  <- U2  spine
    out[:, :, 3, :] = u[:, :, 3, :]   # N4  <- U4  head
    out[:, :, 2, :] = u[:, :, 2, :]   # N3  <- U3  neck

    # mapping: left arm
    out[:, :, 4, :] = u[:, :, 4, :]   # N5  <- U5  left shoulder
    out[:, :, 5, :] = u[:, :, 5, :]   # N6  <- U6  left elbow
    out[:, :, 6, :] = u[:, :, 6, :]   # N7  <- U7  left wrist
    out[:, :, 7, :] = u[:, :, 7, :]   # N8  <- U8  left hand

    # mapping: right arm
    out[:, :, 8, :]  = u[:, :, 8, :]   # N9  <- U9  right shoulder
    out[:, :, 9, :]  = u[:, :, 9, :]   # N10 <- U10 right elbow
    out[:, :, 10, :] = u[:, :, 10, :]  # N11 <- U11 right wrist
    out[:, :, 11, :] = u[:, :, 11, :]  # N12 <- U12 right hand

    # mapping: left leg
    out[:, :, 12, :] = u[:, :, 12, :]  # N13 <- U13 left hip
    out[:, :, 13, :] = u[:, :, 13, :]  # N14 <- U14 left knee
    out[:, :, 14, :] = u[:, :, 14, :]  # N15 <- U15 left ankle
    out[:, :, 15, :] = u[:, :, 15, :]  # N16 <- U16 left foot

    # mapping: right leg
    out[:, :, 16, :] = u[:, :, 16, :]  # N17 <- U17 right hip
    out[:, :, 17, :] = u[:, :, 17, :]  # N18 <- U18 right knee
    out[:, :, 18, :] = u[:, :, 18, :]  # N19 <- U19 right ankle
    out[:, :, 19, :] = u[:, :, 19, :]  # N20 <- U20 right foot

    # approximate extra NTU joints by reusing some UCLA joints
    out[:, :, 20, :] = u[:, :, 2, :]   # N21 spineShoulder <- U3
    out[:, :, 21, :] = u[:, :, 7, :]   # N22 leftHandTip   <- U8
    out[:, :, 22, :] = u[:, :, 7, :]   # N23 leftThumb     <- U8
    out[:, :, 23, :] = u[:, :, 11, :]  # N24 rightHandTip  <- U12
    out[:, :, 24, :] = u[:, :, 11, :]  # N25 rightThumb    <- U12

    return out




@torch.no_grad()
def extract_feats_ucla(
        dataloader: torch.utils.data.DataLoader, 
        model_wrapper:CascadeFormerWrapper, 
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    
    all_feats = []
    all_labels = []

    model_wrapper.t1.eval()
    model_wrapper.t2.eval()
    model_wrapper.cross_attn.eval()
    model_wrapper.gait_head.eval()

    for batch in tqdm(dataloader, desc="Extracting UCLA features"):
        batch_input, batch_labels = batch[0], batch[1]

        # NOTE: make sure that data shape is aligned (B, T, J=25, C=3)
        batch_input = ucla20_to_ntu25(batch_input)
        batch_input = batch_input.to(device)
        batch_labels = batch_labels.to(device)
        out = model_wrapper.infer(batch_input)
        emb = torch.from_numpy(out["embedding"])
        all_feats.append(emb)
        all_labels.append(batch_labels.cpu())

    all_feats = torch.cat(all_feats, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_feats, all_labels

def oneshot_nn_eval(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    num_runs: int = 10,
    seed: int = 42,
):
    """
    Training-free 1-shot eval using cosine nearest neighbor.

    train_feats: [N_train, D]
    train_labels: [N_train]
    test_feats:  [N_test, D]
    test_labels: [N_test]
    """
    rng = np.random.default_rng(seed)
    
    train_feats = F.normalize(train_feats, dim=-1)
    test_feats  = F.normalize(test_feats, dim=-1)

    train_feats = train_feats.to(torch.float32)
    test_feats = test_feats.to(torch.float32)

    # mapping: class -> indices in train set
    class_to_indices = {
        c: torch.nonzero(train_labels == c, as_tuple=True)[0]
        for c in range(num_classes)
    }

    for c, idxs in class_to_indices.items():
        if len(idxs) == 0:
            raise ValueError(f"No training samples for class {c} in UCLA train split.")

    accs = []

    for _ in range(num_runs):
        # 1-shot: pick 1 support per class
        support_idxs = []
        support_labels = []
        for c in range(num_classes):
            idxs = class_to_indices[c].numpy()
            chosen = rng.choice(idxs)
            support_idxs.append(chosen)
            support_labels.append(c)

        support_idxs = torch.tensor(support_idxs, dtype=torch.long)
        support_feats = train_feats[support_idxs]          # [C, D]
        support_labels = torch.tensor(support_labels)      # [C]

        # cosine sim: [N_test, D] @ [D, C] = [N_test, C]
        sims = test_feats @ support_feats.T
        pred_idx = sims.argmax(dim=1)                      # [N_test]
        preds = support_labels[pred_idx]                   # [N_test]

        acc = (preds == test_labels).float().mean().item()
        accs.append(acc)

    return float(np.mean(accs)), float(np.std(accs))


def kshot_proto_eval(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    k_shot: int = 1,
    num_runs: int = 10,
    seed: int = 42,
    normalize: bool = True,
):
    """
    Training-free k-shot eval with prototypical classifier.

    train_feats: [N_train, D]
    train_labels: [N_train]
    test_feats:  [N_test, D]
    test_labels: [N_test]
    """
    rng = np.random.default_rng(seed)

    train_feats = train_feats.to(torch.float32)
    test_feats  = test_feats.to(torch.float32)

    train_feats = F.normalize(train_feats, dim=-1)
    test_feats  = F.normalize(test_feats, dim=-1)

    # mapping: class -> indices in train set
    class_to_indices = {
        c: torch.nonzero(train_labels == c, as_tuple=True)[0]
        for c in range(num_classes)
    }

    for c, idxs in class_to_indices.items():
        if len(idxs) == 0:
            raise ValueError(f"No training samples for class {c} in UCLA train split.")

    accs = []

    for _ in range(num_runs):
        proto_list = []
        proto_labels = []

        for c in range(num_classes):
            idxs = class_to_indices[c].numpy()
            # sample k_shot indices for this class
            if len(idxs) >= k_shot:
                chosen = rng.choice(idxs, size=k_shot, replace=False)
            else:
                # if a class has fewer than k examples, sample with replacement
                chosen = rng.choice(idxs, size=k_shot, replace=True)

            chosen = torch.tensor(chosen, dtype=torch.long)
            support_feats_c = train_feats[chosen]        # [k_shot, D]
            prototype_c = support_feats_c.mean(dim=0)    # [D]

            proto_list.append(prototype_c)
            proto_labels.append(c)

        prototypes = torch.stack(proto_list, dim=0)      # [C, D]
        proto_labels = torch.tensor(proto_labels)        # [C]

        # cosine sim via dot product (embeddings already normalized if normalize=True)
        sims = test_feats @ prototypes.T                # [N_test, C]
        pred_idx = sims.argmax(dim=1)
        preds = proto_labels[pred_idx]

        acc = (preds == test_labels).float().mean().item()
        accs.append(acc)

    return float(np.mean(accs)), float(np.std(accs))


def main():
    args = parse_args()
    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"[INFO] NTU -> NW-UCLA 1-shot eval (training-free) with CascadeFormer on {device}")
    print("=" * 60)
    data_path = "N-UCLA_processed/"
    train_label_path = data_path + "train_label.pkl"
    test_label_path  = data_path + "val_label.pkl"

    # no crazy augmentation for eval
    train_dataset_pre = SF_UCLA_Dataset(
        data_path=data_path,
        label_path=train_label_path,
        data_type="j",
        window_size=-1,
        partition=True,
        repeat=1,
        p=0.0,
        debug=False,
    )

    test_dataset_pre = SF_UCLA_Dataset(
        data_path=data_path,
        label_path=test_label_path,
        data_type="j",
        window_size=-1,
        partition=True,
        repeat=1,
        p=0.0,
        debug=False,
    )

    train_seq, train_lbl = [], []
    for i in range(len(train_dataset_pre)):
        data, _, label, _ = train_dataset_pre[i]
        train_seq.append(torch.from_numpy(data))
        train_lbl.append(label)

    test_seq, test_lbl = [], []
    for i in range(len(test_dataset_pre)):
        data, _, label, _ = test_dataset_pre[i]
        test_seq.append(torch.from_numpy(data))
        test_lbl.append(label)

    num_classes = max(train_lbl + test_lbl) + 1
    print(f"[INFO] UCLA num_classes: {num_classes}")

    train_dataset = ActionRecognitionDataset(train_seq, train_lbl)
    test_dataset  = ActionRecognitionDataset(test_seq,  test_lbl)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=skateformer_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=skateformer_collate_fn,
    )

    model_wrapper = CascadeFormerWrapper(device=device)

    # 3) Extract UCLA train/test embeddings from frozen NTU model
    print("[INFO] Extracting train features...")
    train_feats, train_labels = extract_feats_ucla(train_loader, model_wrapper, device)
    print("[INFO] Extracting test features...")
    test_feats, test_labels   = extract_feats_ucla(test_loader, model_wrapper, device)

    # 4) Run 1-shot nearest neighbor eval
    mean_acc, std_acc = oneshot_nn_eval(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        num_classes=num_classes,
        num_runs=args.num_runs,
    )

    print("=" * 60)
    print(f"[RESULT] NTU -> NW-UCLA 1-shot (training-free, CascadeFormer embedding): "
          f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print("=" * 60)

    # 5-shot
    mean_acc, std_acc = kshot_proto_eval(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        num_classes=num_classes,
        num_runs=args.num_runs,
        k_shot=5,
    )
    print("=" * 60)
    print(f"[RESULT] NTU -> NW-UCLA 5-shot (training-free, CascadeFormer embedding): "
          f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()





