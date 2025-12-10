from collections import OrderedDict
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import random
import os
import argparse
from torch.utils.data import DataLoader
from base_dataset import ActionRecognitionDataset
from SF_UCLA_loader import SF_UCLA_Dataset, skateformer_collate_fn
from MB_utils import ActionNet
from DSTFormer import load_backbone

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



def ucla20_to_mb17_single(skel_batch: torch.Tensor) -> torch.Tensor:
    """
    skel_batch:
        (B, T, 60) or (B, T, 20, 3)
        UCLA 20-joint skeletons (x,y,z or similar).
    returns:
        (B, 1, T, 17, 3) for MotionBERT ActionNet:
        1 person, 17 joints, channels=(x,y,score).
    """
    if skel_batch.dim() == 3:
        B, T, F = skel_batch.shape
        assert F == 60, f"Expected 60 features (20*3), got {F}"
        skel_batch = skel_batch.view(B, T, 20, 3)
    elif skel_batch.dim() == 4:
        B, T, J, C = skel_batch.shape
        assert J == 20 and C >= 2, f"Expected (20,>=2), got ({J},{C})"
    else:
        raise ValueError(f"Unexpected skel_batch shape {skel_batch.shape}")

    B, T, J, C = skel_batch.shape
    device = skel_batch.device
    dtype  = skel_batch.dtype

    u = skel_batch  # (B, T, 20, 3)

    # Build 17-joint array (x,y,score)
    out = torch.zeros(B, T, 17, 3, device=device, dtype=dtype)

    # use first two channels as x,y
    xy = u[..., :2]  # (B,T,20,2)

    # map Kinect joints (U#) to COCO joints (0..16)
    # head index = 3 (U4)
    head = xy[:, :, 3, :]

    # 0-4: face joints all approximated by head
    out[:, :, 0, :2] = head  # nose
    out[:, :, 1, :2] = head  # left eye
    out[:, :, 2, :2] = head  # right eye
    out[:, :, 3, :2] = head  # left ear
    out[:, :, 4, :2] = head  # right ear

    # shoulders, elbows, wrists
    out[:, :, 5, :2] = xy[:, :, 4, :]   # l_shoulder  <- U5
    out[:, :, 6, :2] = xy[:, :, 8, :]   # r_shoulder  <- U9
    out[:, :, 7, :2] = xy[:, :, 5, :]   # l_elbow     <- U6
    out[:, :, 8, :2] = xy[:, :, 9, :]   # r_elbow     <- U10
    out[:, :, 9, :2] = xy[:, :, 6, :]   # l_wrist     <- U7
    out[:, :, 10, :2] = xy[:, :, 10, :] # r_wrist     <- U11

    # hips, knees, ankles
    out[:, :, 11, :2] = xy[:, :, 12, :] # l_hip   <- U13
    out[:, :, 12, :2] = xy[:, :, 16, :] # r_hip   <- U17
    out[:, :, 13, :2] = xy[:, :, 13, :] # l_knee  <- U14
    out[:, :, 14, :2] = xy[:, :, 17, :] # r_knee  <- U18
    out[:, :, 15, :2] = xy[:, :, 14, :] # l_ankle <- U15
    out[:, :, 16, :2] = xy[:, :, 18, :] # r_ankle <- U19

    # set score = 1.0 for all joints
    out[:, :, :, 2] = 1.0

    # add person dim M=1 → (B,1,T,17,3)
    out = out.unsqueeze(1)
    return out

def extract_feats(dataloader_x, model, device):
    all_feats = []
    all_gts = []
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in tqdm(enumerate(dataloader_x)):    # (N, 2, T, 17, 3)
            batch_input = ucla20_to_mb17_single(batch_input)
            batch_input = batch_input.to(device)
            batch_gt = batch_gt.to(device)
            feat = model(batch_input)
            all_feats.append(feat)
            all_gts.append(batch_gt)
    all_feats = torch.cat(all_feats)
    all_gts = torch.cat(all_gts)
    return all_feats, all_gts

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
            idxs = class_to_indices[c].cpu().numpy()
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
    print(f"[INFO] NTU -> NW-UCLA 1-shot eval (training-free) with MotionBert on {device}")
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

    chk_filename = os.path.join("action_checkpoints/MotionBert", "best_epoch.bin")
    print('Loading backbone', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    raw_state: OrderedDict = checkpoint["model"]

    # Build a state dict only for the MotionBert backbone
    backbone_state = OrderedDict()
    for k, v in raw_state.items():
        # strip "module"
        if k.startswith("module.backbone."):
            k = k[len("module.backbone."):]
            backbone_state[k] = v

    # print(backbone_state.keys())

    model_backbone = load_backbone()  # this should create a DSTformer
    model_backbone.load_state_dict(backbone_state, strict=True)
    model_wrapper = ActionNet(
        backbone=model_backbone,
        dim_rep=512, 
        dropout_ratio=0.5, 
        version='embed',      # 🔥 use embed head for feature extraction (NOT a classification head)
        hidden_dim=2048, 
        num_joints=17
    ).to(device)
    model_wrapper.eval()

    print("[INFO] Extracting train features...")
    train_feats, train_labels = extract_feats(train_loader, model_wrapper, device)
    print("[INFO] Extracting test features...")
    test_feats, test_labels = extract_feats(test_loader, model_wrapper, device)
    train_feats = train_feats.cpu()
    train_labels = train_labels.cpu()
    test_feats  = test_feats.cpu()
    test_labels = test_labels.cpu()

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
    print(f"[RESULT] NTU -> NW-UCLA 1-shot (training-free, MotionBert embedding): "
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
    print(f"[RESULT] NTU -> NW-UCLA 5-shot (training-free, MotionBert embedding): "
          f"{mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()





