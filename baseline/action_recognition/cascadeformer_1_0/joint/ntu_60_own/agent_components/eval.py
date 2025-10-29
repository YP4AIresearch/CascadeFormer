from typing import Any, Dict, List
import json
import torch
from torch.utils.data import DataLoader
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tqdm
from NTU_feeder import Feeder
from .statistics import DistanceScorer, score_anomaly, perceive_window
from .perceiver import CascadeFormerWrapper
from .rag import retrieve_context
from .constants import policy_chain, DATA_PATH, WINDOW_SIZE
from .data_utils import select_main_person_batch
from .runner import is_abnormal_label


def decide_without_log(policies_store, incidents_store, event: Dict[str, Any], scores: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same as decide(), but does NOT modify the incidents_store or append new entries.
    Useful for offline evaluation (no side effects).
    """
    ctx, q = retrieve_context(policies_store, incidents_store, event, scores)
    out = policy_chain.invoke({"event": event, "scores": scores, "context": ctx})

    try:
        # Remove possible Markdown code fences (```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.MULTILINE)
        decision = json.loads(cleaned)
    except Exception as e:
        # Silent fallback to "LOG" to keep evaluation consistent
        decision = {"action": "LOG", "rationale": f"fallback: {str(e)}"}
    return decision


def process_window_without_log(policies_store, incidents_store, knn: DistanceScorer, model: CascadeFormerWrapper, skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # 3) Decide (your normal function)
    decision = decide_without_log(policies_store, incidents_store, event, scores)

    return {"event": event, "scores": scores, "decision": decision}

def evaluate_full_test_split_with_agent(policies_store, incidents_store, knn_scorer, model: CascadeFormerWrapper,
                                        batch_size: int = 16, device: str = "cuda"):
    """
    Loops over the test split and calls your existing process_window(...) per sample.
    Metrics are computed for:
      y_true: 1 if GT label ∈ abnormal_action_labels else 0
      y_pred: 1 if decision['action'] == 'ALERT' else 0
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=False,
        p_interval=[0.5, 1],
        vel=False,
        bone=False
    )
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred = [], []

    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(loader):
            skeletons = skeletons.to(device)                 # (B,C,T,V,M)
            labels_np = labels.cpu().numpy().astype(int)     # (B,)
            windows = select_main_person_batch(skeletons)   # (B,T,V,C)

            # Iterate samples in this batch and reuse the full agent path
            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)  # (T,V,C)
                result = process_window_without_log(
                    policies_store, incidents_store,
                    knn=knn_scorer, model=model,
                    skel_window=window_np.tolist()
                )

                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn   = 1 if is_abnormal_label(int(labels_np[i])) else 0

                y_pred.append(pred_alert)
                y_true.append(true_abn)
            
            break # for quick demo, remove this line to run full test split

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("\n=== Offline Evaluation (Agent over TEST split) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)
    print("Confusion Matrix (rows=truth [Normal, Abnormal]; cols=pred [LOG, ALERT])", flush=True)
    print(cm, flush=True)


def evaluate_random_batches_with_agent(policies_store, incidents_store, knn_scorer, model: CascadeFormerWrapper,
                                       num_batches: int = 10, batch_size: int = 16, device: str = "cuda"):
    """
    Randomly samples `num_batches` batches from the test split (each of size `batch_size`).
    Runs the full agent inference path and computes classification metrics.
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split='test',
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=False,
        p_interval=[0.5, 1],
        vel=False,
        bone=False
    )
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)
    
    # --- randomly select `num_batches` distinct indices ---
    import random
    seed = 42
    random.seed(seed)
    selected_batches = sorted(random.sample(range(total_batches), min(num_batches, total_batches)))
    print(f"[Info] Evaluating {len(selected_batches)} random batches out of {total_batches} total.")

    y_true, y_pred = [], []

    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    with torch.inference_mode():
        for b_idx, (skeletons, labels, _) in enumerate(loader):
            if b_idx not in selected_batches:
                continue  # skip batches not selected

            skeletons = skeletons.to(device)                 
            labels_np = labels.cpu().numpy().astype(int)     
            windows = select_main_person_batch(skeletons)   # (B,T,V,C)

            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)
                result = process_window_without_log(
                    policies_store, incidents_store,
                    knn=knn_scorer, model=model,
                    skel_window=window_np.tolist()
                )

                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn   = 1 if is_abnormal_label(int(labels_np[i])) else 0

                y_pred.append(pred_alert)
                y_true.append(true_abn)

    # --- compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    print("\n=== Offline Evaluation (Agent over TEST split) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)
    print("Confusion Matrix (rows=truth [Normal, Abnormal]; cols=pred [LOG, ALERT])", flush=True)
    print(cm, flush=True)