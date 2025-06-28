
import numpy as np
import torch
import argparse
from torch import nn
from base_dataset import ActionRecognitionDataset
from penn_utils import set_seed
from NTU_utils import NUM_JOINTS_NTU
from finetuning import load_T1, load_T2, load_cross_attn, GaitRecognitionHead
from torch_lr_finder import LRFinder

class CascadeWrapper(nn.Module):
    def __init__(self, T1, T2, cross_attn, head):
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.cross_attn = cross_attn
        self.head = head

    def forward(self, x):
        # Assume input: (B, T, J, 3)
        feat1 = self.T1(x)                # (B, T, D)
        feat2 = self.T2(x)                # (B, T, D)
        fused = self.cross_attn(feat1, feat2)
        out = self.head(fused)
        return out


def load_cached_data(path="ntu_cache_train_sub.npz"):
    data = np.load(path, allow_pickle=True)
    sequences = list(data["sequences"])
    labels = list(data["labels"])
    return sequences, labels

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Inference")
    parser.add_argument("--root_dir", type=str, default="", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for Inference")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)

    args = parse_args()
    # get the number of classes from the root_dir by taking the trailing number
    batch_size = args.batch_size
    device = args.device

    # Set the device

    hidden_size = 512 # 256, 512, 768, 1024
    n_heads = 8
    num_layers = 8    # 4, 8, 12
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 50)
    print(f"[INFO] Starting NTU dataset processing on {device}...")
    print("=" * 50)

    # load the dataset
    test_seq, test_lbl = load_cached_data('ntu_cache_test_sub_64_10.npz')    
    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)
    
    # get the number of classes
    num_classes = len(set(test_lbl))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

    # load T1 model
    unfreeze_layers = "entire"
    if unfreeze_layers is None:
        print("************Freezing all layers")
        t1 = load_T1("action_checkpoints/fixed_ntu/NTU_pretrained.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
    else:
        t1 = load_T1("action_checkpoints/fixed_ntu/NTU_finetuned_T1.pt", d_model=hidden_size, num_joints=NUM_JOINTS_NTU, three_d=True, nhead=n_heads, num_layers=num_layers, device=device)
        print(f"************Unfreezing layers: {unfreeze_layers}")
    
    t2 = load_T2("action_checkpoints/fixed_ntu/NTU_finetuned_T2.pt", d_model=hidden_size, nhead=n_heads, num_layers=num_layers, device=device)
    # load the cross attention module
    cross_attn = load_cross_attn("action_checkpoints/fixed_ntu/NTU_finetuned_cross_attn.pt", d_model=hidden_size, device=device)

    # load the gait recognition head
    gait_head = GaitRecognitionHead(input_dim=hidden_size, num_classes=num_classes)
    gait_head.load_state_dict(torch.load("action_checkpoints/fixed_ntu/NTU_finetuned_head.pt", map_location="cpu"))
    gait_head = gait_head.to(device)

    print("Aha! All models loaded successfully!")
    print("=" * 100)

    # evaluate the model
    print("=" * 50)
    print("[INFO] Starting LR search...")
    print("=" * 50)

    # Initialize LR Finder
    model = CascadeWrapper(t1, t2, cross_attn, gait_head).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Initial learning rate
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(test_loader, end_lr=1, num_iter=100, step_mode="exp")
    lr_finder.plot()  # Plot the learning rate vs loss
    # save the plot
    lr_finder.save_plot("lr_finder_plot.png")




if __name__ == "__main__":
    main()