import os
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import joblib
import pandas as pd
import matplotlib
import random
from pathlib import Path
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from NTU_feeder import Feeder
from dotenv import load_dotenv
from agent_components.constants import WINDOW_SIZE, DATA_PATH, ST_RE
from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer, score_anomaly, perceive_window
from agent_components.rag import print_incident_db, print_policy_db
from agent_components.reinforcement import PolicyParams, train_a_reward_model, policy_search, decide_with_rl_policy
from agent_components.runner import is_abnormal_label
from agent_components.data_utils import extract_state_features, select_main_person_batch

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/peng.1007/CascadeFormer/.env")
# Access your key
api_key = os.getenv("OPENAI_KEY")


def process_window_RL(
    policies_store, incidents_store,
    knn: DistanceScorer, model: CascadeFormerWrapper,
    skel_window: List[List[List[float]]],
    learned_params: PolicyParams,
    prefer_kb: bool = False,   # ✅ True = favor KB when KB and RL disagree
) -> Dict[str, Any]:
    """
    Hybrid inference that combines KB-based decision and RL-based decision.
    If prefer_kb=True, KB decision takes precedence on disagreement (especially for ALERT cases).
    """
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly (must provide knn_dist / mahalanobis / top1_conf)
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # RL decision
    s = extract_state_features(event, scores)
    rl_action = decide_with_rl_policy(s, learned_params)["action"].upper()

    action = rl_action
    decision = {"action": action, "rationale": ""}
    return {"event": event, "scores": scores, "decision": decision}


def evaluate_random_batches_with_rl_policy(
    policies_store,
    incidents_store,
    knn_scorer,
    model: CascadeFormerWrapper,
    learned_params: PolicyParams,
    num_batches,
    batch_size,
    device: str = "cuda"
):
    """
    Randomly samples `num_batches` batches from the test split.
    Runs the RL policy on each sample and computes classification metrics.
    """
    test_dataset = Feeder(
        data_path=DATA_PATH,
        split="test",
        debug=False,
        random_choose=False,
        random_shift=False,
        random_move=False,
        window_size=WINDOW_SIZE,
        normalization=False,
        random_rot=False,
        p_interval=[0.5, 1],
        vel=False,
        bone=False,
    )
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    total_batches = len(loader)

    # --- 2. Randomly choose a subset of batches ---
    seed = 42
    random.seed(seed)
    selected_batches = sorted(random.sample(range(total_batches), min(num_batches, total_batches)))
    print(f"[INFO] Evaluating {len(selected_batches)} random batches out of {total_batches} total.", flush=True)

    # --- 3. Metric containers ---
    y_true, y_pred = [], []

    # --- 4. Set model to eval mode ---
    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    # --- 5. Evaluation loop ---
    with torch.inference_mode():
        for b_idx, (skeletons, labels, _) in enumerate(loader):
            if b_idx not in selected_batches:
                continue  # skip non-selected batches

            skeletons = skeletons.to(device)
            labels_np = labels.cpu().numpy().astype(int)
            windows = select_main_person_batch(skeletons)

            for i in range(windows.shape[0]):
                window_np = windows[i].cpu().numpy().astype(float)
                result = process_window_RL(
                    policies_store,
                    incidents_store,
                    knn=knn_scorer,
                    model=model,
                    skel_window=window_np.tolist(),
                    learned_params=learned_params,
                )

                pred_alert = 1 if str(result["decision"]["action"]).upper() == "ALERT" else 0
                true_abn = 1 if is_abnormal_label(int(labels_np[i])) else 0
                
                # print("===========", flush=True)
                # print(f"instance # {b_idx * batch_size + i}:", flush=True)
                # print(f"action: {result['decision']['action']}", flush=True)
                # print("-----------", flush=True)

                y_pred.append(pred_alert)
                y_true.append(true_abn)

    # --- 6. Compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # --- 7. Print results ---
    print("\n=== Offline Evaluation (RL Policy over RANDOM TEST Batches) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)


def rl_policy_optimization(incidents_df: pd.DataFrame,
                                policies_store, incidents_store,
                                knn_scorer, model: CascadeFormerWrapper,
                                device: str = "cuda"):
    # 1) Train reward model from past incidents
    r_model = train_a_reward_model(incidents_df)

    # 2) Search best policy params under learned R(s,a) reward model
    best_params, best_reward = policy_search(r_model, incidents_df)
    print("\n=== RL-based Policy Optimization Result ===", flush=True)
    print("[RL] Best params:", best_params)
    print("[RL] Highest reward:", best_reward)
    print("===========================================", flush=True)

    # 3) Evaluate on held-out TEST split using learned policy
    evaluate_random_batches_with_rl_policy(
        policies_store, incidents_store, knn_scorer, model,
        learned_params=best_params, 
        num_batches=200,
        batch_size=1,
        device=device
    )

    # FIXME: (Optional) Persist best policy into your policies_store as a structured rule
    # so the production agent can cite it:
    # add_policy_to_store(policies_store, best_params)  # implement if you want provenance


def agent_rl_policy_optimization():
    """
    RL-based policy optimization entrypoint.
    Builds incidents_df from KB, trains reward model, searches best policy.
    """
    model = CascadeFormerWrapper(device="cuda")
    
    # ---- Load or build KNN scorer ----
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        joblib.dump(knn, "trained_knn.pkl")
    knn = joblib.load("trained_knn.pkl")

    # ---- Load vector stores ----
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    policies_store  = FAISS.load_local("vectorstores/policies",  emb, allow_dangerous_deserialization=True)
    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)

    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    def _parse_incidents_kb_file(path: str = "incidents_db.kb") -> pd.DataFrame:
        p = Path(path)
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("DUMMY") or set(s) == {"-"}:
                continue
            m = ST_RE.search(s)
            d = m.groupdict()
            rows.append({
                "entropy": float(d["entropy"]),
                "knn_dist": float(d["knn_dist"]),
                "mahalanobis": float(d["mahalanobis"]),
                "top1_conf": float(d["top1_conf"]),
                "decision": d["decision"].upper(),
                "gt_label": (d["gt_label"]).lower(),
            })
        return pd.DataFrame(rows)

    incidents_df = _parse_incidents_kb_file("incidents_db.kb")
    print(f"[incidents_df] {len(incidents_df)} rows | columns: {list(incidents_df.columns)}")

    rl_policy_optimization(
        incidents_df,
        policies_store, incidents_store,
        knn, model,
        device="cuda"
    )


if __name__ == "__main__":

    # mode 1: running agent training (optional) + a random inference demo
    #agent_training_and_demo(inference_only=True)


    # mode 2: reinforcement-learning based policy optimization
    agent_rl_policy_optimization()


