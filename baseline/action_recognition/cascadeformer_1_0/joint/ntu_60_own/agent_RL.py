import os
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import joblib
import pandas as pd
import matplotlib
from tqdm import tqdm
from pathlib import Path
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from NTU_feeder import Feeder
from dotenv import load_dotenv
from agent_components.constants import WINDOW_SIZE, DATA_PATH, abnormal_action_labels, ST_RE
from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer, score_anomaly, perceive_window
from agent_components.rag import write_incident_db, print_incident_db, write_policy_db, print_policy_db
from agent_components.reinforcement import PolicyParams, train_a_reward_model, random_policy_search, decide_with_rl_policy
from agent_components.runner import is_abnormal_label, train_one_sample, inference_demo
from agent_components.data_utils import extract_state_features, select_main_person_batch
from agent_components.eval import decide_without_log

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/peng.1007/CascadeFormer/.env")
# Access your key
api_key = os.getenv("OPENAI_KEY")


def agent_training_and_demo(inference_only: bool):
    INFERENCE_ONLY = inference_only
    n_samples = 10
    model = CascadeFormerWrapper(device="cuda")
    
    # if no trained knn, create one
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        # save KNN model for later use
        joblib.dump(knn, "trained_knn.pkl")

    # load the KNN
    knn = joblib.load("trained_knn.pkl")
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    if not INFERENCE_ONLY:
        # build new vector stores
        policies_store = FAISS.from_texts(
            texts=[
                f"Raise an ALERT if the predicted action is within the abnormal action list "f"({', '.join(abnormal_action_labels)})."
            ],
            embedding=emb
        )

        incidents_store = FAISS.from_texts(
            texts=[
                "DUMMY INCIDENT ENTRY; DO NOT USE.",
            ],
            embedding=emb
        )
    else:
        # load existing vector stores
        policies_store = FAISS.load_local("vectorstores/policies", emb, allow_dangerous_deserialization=True)
        incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)


    if not INFERENCE_ONLY:
        # training code
        print(f"Start training with {n_samples} samples...", flush=True)
        for i in range(n_samples):
            print(f"\n=== Training iteration {i+1}/{n_samples} ===", flush=True)
            train_one_sample(policies_store, incidents_store, model, knn, json_path="demo_window.json")

        os.makedirs("vectorstores", exist_ok=True)
        policies_store.save_local("vectorstores/policies")
        incidents_store.save_local("vectorstores/incidents")

        # write the knowledge base to text files
        # NOTE: IMPORTANT - only write when training, not during inference
        write_incident_db(incidents_store)
        write_policy_db(policies_store)


    # print both policies and incidents
    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    print("\n\nStart an inference demo...", flush=True)
    DEMO_VIDEO_PATH = "demo_window.mp4"
    DEMO_JSON_PATH = "demo_window.json"
    policies_store = FAISS.load_local("vectorstores/policies", emb, allow_dangerous_deserialization=True)
    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)

    inference_demo(policies_store, incidents_store, model, knn, json_path=DEMO_JSON_PATH, video_path=DEMO_VIDEO_PATH)




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

    # KB decision
    kb_action = decide_without_log(policies_store, incidents_store, event, scores)["action"].upper()

    # RL decision
    s = extract_state_features(event, scores)
    rl_action = decide_with_rl_policy(s, learned_params)["action"].upper()

    # resolve the final action: who overrides whom?
    if kb_action == rl_action:
        action = kb_action
        rationale = f"KB and RL agree: {action}."
    else:
        if prefer_kb:
            action = kb_action
            rationale = f"Disagreement: KB={kb_action}, RL={rl_action}. KB preferred, so {action}."
        else:
            action = rl_action
            rationale = f"Disagreement: KB={kb_action}, RL={rl_action}. RL preferred, so {action}."

    decision = {"action": action, "rationale": rationale}
    return {"event": event, "scores": scores, "decision": decision}


def evaluate_random_batches_with_rl_policy(
    policies_store,
    incidents_store,
    knn_scorer,
    model: CascadeFormerWrapper,
    learned_params: PolicyParams,
    num_batches: int = 10, # now 10 random batches by default
    batch_size: int = 16,
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
    import random
    selected_batches = sorted(random.sample(range(total_batches), min(num_batches, total_batches)))
    print(f"[Info] Evaluating {len(selected_batches)} random batches out of {total_batches} total.", flush=True)

    # --- 3. Metric containers ---
    y_true, y_pred = [], []

    # --- 4. Set model to eval mode ---
    model.t1.eval()
    model.t2.eval()
    model.cross_attn.eval()
    model.gait_head.eval()

    # --- 5. Evaluation loop ---
    with torch.inference_mode():
        for b_idx, (skeletons, labels, _) in tqdm(enumerate(loader)):
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

                y_pred.append(pred_alert)
                y_true.append(true_abn)

    # --- 6. Compute metrics ---
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # --- 7. Print results ---
    print("\n=== Offline Evaluation (RL Policy over RANDOM TEST Batches) ===", flush=True)
    print(f"Samples   : {len(y_true)}", flush=True)
    print(f"Accuracy  : {acc:.4f}", flush=True)
    print(f"Precision : {prec:.4f}", flush=True)
    print(f"Recall    : {rec:.4f}", flush=True)
    print(f"F1-score  : {f1:.4f}", flush=True)
    print("Confusion Matrix (rows=truth [Normal, Abnormal]; cols=pred [LOG, ALERT])", flush=True)
    print(cm, flush=True)


def rl_policy_optimization(incidents_df: pd.DataFrame,
                                policies_store, incidents_store,
                                knn_scorer, model: CascadeFormerWrapper,
                                device: str = "cuda"):
    # 1) Train reward model from past incidents
    r_model = train_a_reward_model(incidents_df)

    # 2) Search best policy params under learned R(s,a) reward model
    best_params = random_policy_search(r_model, incidents_df, rng=42)
    print("\n=== RL-based Policy Optimization Result ===", flush=True)
    print("[RL] Best params:", best_params)
    print("\n===========================================", flush=True)

    return

    # 3) Evaluate on held-out TEST split using learned policy
    evaluate_random_batches_with_rl_policy(
        policies_store, incidents_store, knn_scorer, model,
        learned_params=best_params, batch_size=16, device=device
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


