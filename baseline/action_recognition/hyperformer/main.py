import torch
import argparse
from base_dataset import ActionRecognitionDataset
from HyperFormer import Model as HyperFormer
from NTU_utils import split_train_val, NUM_JOINTS_NTU
from penn_utils import set_seed
import torch.optim as optim
import torch.nn.functional as F
from tqdm import trange, tqdm

def load_cached_data(path="ntu_train_data.pt"):
    data = torch.load(path)
    return data['sequences'], data['labels'].tolist() 

def parse_args():
    parser = argparse.ArgumentParser(description="Gait Recognition Training")
    parser.add_argument("--train", action='store_true', help="Run the stage of pretraining")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for the model")
    parser.add_argument("--class_specific_split", action='store_true', help="Use class-specific split for training and validation")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training (cuda or cpu)")
    return parser.parse_args()


def main():
    set_seed(42)
    batch_size = 64 
    num_classes = 60  # NTU has 60 classes
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    JOINT_LABELS = [0, 4, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 1, 0, 1, 0, 1]
    # FIXME: I am using single person action recognition FOR NOW
    NUM_PERSONS = 1

    args = parse_args()
    TRAIN = args.train

    if TRAIN:
        print("=" * 50)
        print(f"[INFO] Starting NTU dataset processing on {device}...")
        print("=" * 50)

        # load the dataset
        import time
        t_start = time.time()
        all_seq_clean, all_lbl_clean = load_cached_data('CORRECTED_ntu_cache_train_sub_64_10_augmented.pt')
        train_seq, train_lbl, val_seq, val_lbl = split_train_val(all_seq_clean, all_lbl_clean, val_ratio=0.20)
        t_end = time.time()
        print(f"[INFO] Time taken to load NTU skeletons: {t_end - t_start:.2f} seconds")        

        train_finetuning_dataset = ActionRecognitionDataset(train_seq, train_lbl)
        val_finetuning_dataset = ActionRecognitionDataset(val_seq, val_lbl)

        train_loader = torch.utils.data.DataLoader(
            train_finetuning_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_finetuning_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        model = HyperFormer(
            num_class=num_classes,
            num_point=NUM_JOINTS_NTU,
            num_person=NUM_PERSONS,
            graph='graph.ntu_rgb_d.Graph',
            graph_args={'labeling_mode': 'spatial'},
            joint_label=JOINT_LABELS,
            in_channels=3,
            drop_out=0.5,
            num_of_heads=9
        ).to(device)

        optimizer = optim.SGD(
            model.parameters(),
            lr=0.025,
            momentum=0.9,
            nesterov=True,
            weight_decay=0.0004
        )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[110, 120],
            gamma=0.1
        )
        num_epochs = 140
        for epoch in trange(num_epochs, desc="Training Progress"):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for x, y in tqdm(train_loader, desc=f"Train [{epoch+1}]"):
                x, y = x.to(device), y.to(device)
                # make sure the input shape matches the model's expectation
                x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, D, T, J, M=1)

                logits, _ = model(x, y)
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * y.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

            train_acc = correct / total
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss / total:.4f} | Acc: {train_acc:.4f}")

            # validation
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Val [{epoch+1}]"):
                    # x: (B, T, J, D)
                    x, y = x.to(device), y.to(device)

                    # make sure the input shape matches the model's expectation
                    x = x.permute(0, 3, 1, 2).unsqueeze(-1)

                    logits, _ = model(x, y)
                    loss = F.cross_entropy(logits, y)
                    val_loss += loss.item() * y.size(0)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss / total:.4f} | Acc: {val_acc:.4f}")

            lr_scheduler.step()

        print("Training completed successfully!")
        # Save the model
        torch.save(model.state_dict(), "action_checkpoints/NTU_hyperformer.pt")
        print("Model saved successfully!")

    else:
        print("no training, just testing...")

    # test set evaluation
    print("=" * 50)
    print(f"[INFO] Starting NTU dataset testing on {device}...")
    print("=" * 50)
    test_seq, test_lbl = load_cached_data('CORRECTED_ntu_cache_test_sub_64_10.pt')
    test_dataset = ActionRecognitionDataset(test_seq, test_lbl)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Load the model
    model = HyperFormer(
        num_class=num_classes,
        num_point=NUM_JOINTS_NTU,
        num_person=NUM_PERSONS,
        graph='graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'spatial'},
        joint_label=JOINT_LABELS,
        in_channels=3,
        drop_out=0.5,
        num_of_heads=9
    ).to(device)
    model.load_state_dict(torch.load("action_checkpoints/NTU_hyperformer.pt"))
    print("[INFO] Model loaded successfully!")

    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, D, T, J, M=1)

            logits, _ = model(x, y)
            loss = F.cross_entropy(logits, y)
            test_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"[Test] Loss: {test_loss / total:.4f} | Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()

