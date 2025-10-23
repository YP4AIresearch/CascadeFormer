import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Union
from typing import Optional
import joblib
import time
import csv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import re
import matplotlib
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from sklearn.neighbors import NearestNeighbors
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# local import
from NTU_feeder import Feeder
from finetuning import load_T1, load_T2, load_cross_attn_with_ffn, GaitRecognitionHeadMLP
from dotenv import load_dotenv

WINDOW_SIZE = 64
DATA_PATH = "NTU60_CS.npz"

NTU25_EDGES = [
    (0, 1), (1, 1), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13),
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (20, 1), (21, 7),
    (22, 7), (23, 11), (24, 11)
]

# Load environment variables from .env file
load_dotenv(dotenv_path="/home/peng.1007/Cascade-LA-Agent/BPMT/.env")

# Access your key
api_key = os.getenv("OPENAI_KEY")



########## Dataset #######################################################################
##########################################################################################


dataset_config = {
    "num_classes": 60  # for NTU
}

NUM_JOINTS_NTU = 25


# we have 60 action classes in total
classify_labels = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", 
    "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)", 
    "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", 
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", 
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up", 
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard", 
    "pointing to something with finger", "taking a selfie", "check time (from watch)", 
    "rub two hands together", "nod head/bow", "shake head", "wipe face", 
    "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", 
    "touch chest (stomachache/heart pain)", "touch back (backache)", 
    "touch neck (neckache)",  "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "hugging other person",
    "giving something to other person",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
    "point finger at the other person",
    "touch other person's pocket"
]


normal_action_labels = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair",
    "drop", "pickup", "throw", "sitting down", "standing up (from sitting position)",
    "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up",
    "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "use a fan (with hand or paper)/feeling warm",
    "pat on back of other person", "hugging other person", "giving something to other person",
    "handshaking", "walking towards each other", "walking apart from each other"
]

abnormal_action_labels = [
    "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition",
    "punching/slapping other person", "kicking other person",
    "pushing other person", "point finger at the other person", "touch other person's pocket"
]


########## Model #########################################################################
##########################################################################################

model_config = {
    "t1_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_T1.pt",
    "t2_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_T2.pt",
    "cross_attn_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_cross_attn.pt",
    "gait_head_ckpt": "action_checkpoints/BEST_NTU_1_0_CS/NTU_finetuned_head.pt",
    "hidden_size": 768, # for NTU/CS
    "n_heads": 16,  # for NTU/CS
    "num_layers": 16,  # for NTU/CS
}


class CascadeFormerWrapper:
    def __init__(self, device="cuda"):
        self.device = device

        self.t1 = load_T1(model_config["t1_ckpt"], d_model=model_config["hidden_size"], num_joints=NUM_JOINTS_NTU, three_d=True, nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)

        self.t2 = load_T2(model_config["t2_ckpt"], d_model=model_config["hidden_size"], nhead=model_config["n_heads"], num_layers=model_config["num_layers"], device=device)
        self.cross_attn = load_cross_attn_with_ffn(model_config["cross_attn_ckpt"], d_model=model_config["hidden_size"], device=device)

        # load the gait recognition head
        self.gait_head = GaitRecognitionHeadMLP(input_dim=model_config["hidden_size"], num_classes=dataset_config["num_classes"])
        self.gait_head.load_state_dict(torch.load(model_config["gait_head_ckpt"], map_location="cpu"))
        self.gait_head = self.gait_head.to(device)

        # set models to evaluation mode
        self.t1.eval()
        self.t2.eval()
        self.cross_attn.eval()
        self.gait_head.eval()


    @torch.inference_mode()
    def infer(self, skel_batch: torch.Tensor) -> Dict[str, Any]:
        """
        skel_batch: (B, T, J, C) float32
        returns dict with logits, probs, embedding
        """
        x1 = self.t1.encode(skel_batch.to(self.device))        
        x2 = self.t2.encode(x1)
        fused = self.cross_attn(x1, x2, x2)
        pooled = fused.mean(dim=1)
        logits = self.gait_head(pooled)
        probs = torch.softmax(logits, dim=-1).float()
        embedding = torch.nn.functional.normalize(pooled, dim=-1)
        return {
            "logits": logits.cpu().numpy(),
            "probs": probs.cpu().numpy(),
            "embedding": embedding.detach().cpu().numpy(),
        }

@tool("perceive_window", return_direct=False)
def perceive_window(model: CascadeFormerWrapper, skel_window: List[List[List[float]]]) -> Dict[str, Any]:
    """
        run CascadeFormer on a single window of skeletons and return structured event with probs, entropy, and embedding.
    """
    x = torch.tensor(skel_window, dtype=torch.float32).unsqueeze(0) # shape: (1,T,J,C)
    out = model.infer(x)
    probs = out["probs"][0]
    embedding = out["embedding"]

    event = {
        "top_label": classify_labels[int(np.argmax(probs))],
        "top_prob": float(np.max(probs)),
        "entropy": entropy(probs),
        "embedding": embedding[0].tolist(),  # convert to list for JSON serialization
    }
    return event

########## anomaly detection tool ########################################################
##########################################################################################


def build_normal_bank(
    model: CascadeFormerWrapper,
    dataset,
    batch_size: int = 32,
    device: str = "cuda",
    per_class: bool = False,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Build a bank of embeddings using ONLY normal actions defined in `normal_action_labels`.

    Returns:
        - If per_class == False:
            np.ndarray of shape (N_normals, D)
        - If per_class == True:
            dict[int, np.ndarray], where keys are NORMAL class IDs
    """
    # Map the normal action names to their class indices once.
    normal_name_to_id = {name: i for i, name in enumerate(classify_labels)}
    normal_class_ids = {normal_name_to_id[name] for name in normal_action_labels}
    # (Optional) sanity: ensure disjoint from abnormal list if needed
    # assert not any(id_ in normal_class_ids for id_ in abnormal_class_ids)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if per_class:
        # Only allocate bins for NORMAL classes
        banks: Dict[int, List[np.ndarray]] = {c: [] for c in sorted(normal_class_ids)}
    else:
        embeddings: List[np.ndarray] = []

    with torch.inference_mode():
        for skeletons, labels, _ in tqdm(loader):
            skeletons = skeletons.to(device)

            # Preprocessing sequences from CTR-GCN-style input
            B, C, T, V, M = skeletons.shape
            sequences = skeletons.permute(0, 2, 3, 1, 4)

            # Select most active person (M=1)
            motion = sequences.abs().sum(dim=(1, 2, 3))  # (B, M)
            main_person_idx = motion.argmax(dim=-1)       # (B,)

            indices = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
            sequences = torch.gather(sequences, dim=4, index=indices).squeeze(-1)  # (B, T, V, C)
            skeletons = sequences.float().to(device)  # (B, T, J, D)

            # Forward pass
            out = model.infer(skeletons)         # dict with "embedding" (B, D) as np.ndarray
            emb_batch = out["embedding"]          # (B, D), numpy array

            # Figure out which items in this batch are NORMAL
            labels_np = labels.detach().cpu().numpy()
            keep_idx = [i for i, lbl in enumerate(labels_np) if int(lbl) in normal_class_ids]
            if not keep_idx:
                continue

            emb_norm = emb_batch[keep_idx]        # (B_norm, D)
            lbl_norm = labels_np[keep_idx]        # (B_norm,)

            if per_class:
                for e, lbl in zip(emb_norm, lbl_norm):
                    banks[int(lbl)].append(e)
            else:
                embeddings.append(emb_norm)

    if per_class:
        # Stack each normal class bank; drop classes with zero samples
        stacked = {
            c: np.stack(v, axis=0).astype(np.float32)
            for c, v in banks.items() if len(v) > 0
        }

        return stacked
    else:
        if len(embeddings) == 0:
            # ; return an empty (0, D) array
            print("❌error: No normal samples found")
            return np.zeros((0, model_config["hidden_size"]), dtype=np.float32)
        return np.concatenate(embeddings, axis=0).astype(np.float32)


class DistanceScorer:
    """
        How far is this embedding z from normal training data?
    """
    def __init__(self, model: CascadeFormerWrapper, k=5):
        self.model = model
        self.k = k

        WINDOW_SIZE = 64

        DATA_PATH = "NTU60_CS.npz"
        train_dataset = Feeder(
            data_path=DATA_PATH,
            split='train',
            debug=False,
            random_choose=False,
            random_shift=False,
            random_move=False,
            window_size=WINDOW_SIZE,
            normalization=False,
            random_rot=True,
            p_interval=[0.5, 1],
            vel=False,
            bone=False
        )

        normal_bank = build_normal_bank(self.model, train_dataset, per_class=False)
        self.nn = NearestNeighbors(n_neighbors=k).fit(normal_bank)
        self.normal_bank = normal_bank

    def score(self, z: np.ndarray) -> float:
        dists, _ = self.nn.kneighbors(z.reshape(1, -1))
        return float(dists.mean())

def entropy(p: np.ndarray) -> float:
    """
        How uncertain is the model's prediction?
    """
    p = np.clip(p, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())


def fit_mahalanobis_params(normal_bank: np.ndarray, reg: float = 1e-5):
    """
    From a bank of normal embeddings (N, D), compute the mean and (regularized) inverse covariance.
    Returns (mean, inv_cov).
    """
    if normal_bank.ndim != 2:
        raise ValueError("normal_bank must be a 2D array of shape (N, D).")
    mu = normal_bank.mean(axis=0)
    # Sample covariance (D, D)
    cov = np.cov(normal_bank, rowvar=False)
    # Tikhonov regularization for stability
    inv_cov = np.linalg.pinv(cov + reg * np.eye(cov.shape[0], dtype=cov.dtype))
    return mu.astype(np.float32), inv_cov.astype(np.float32)

def mahalanobis_distance(z: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
    """
    Mahalanobis distance of embedding z to a Gaussian fit (mean, inv_cov) from normal data.
    """
    z = z.astype(np.float32)
    mean = mean.astype(np.float32)
    inv_cov = inv_cov.astype(np.float32)
    diff = z - mean
    # sqrt( (z - μ)^T Σ^{-1} (z - μ) )
    return float(np.sqrt(diff @ inv_cov @ diff))

@tool("score_anomaly", return_direct=False)
def score_anomaly(knn: DistanceScorer, event: Dict[str, Any]) -> Dict[str, Any]:
    """
        compute anomaly scores from embedding and additional signals.
    """
    z = np.array(event["embedding"], dtype=np.float32)
    # compute Mahalanobis distance
    mu, inv_cov = fit_mahalanobis_params(knn.normal_bank)

    scores = {
        "knn_dist": knn.score(z),
        "mahalanobis": mahalanobis_distance(z, mu, inv_cov),
        "ent": event["entropy"],
        "top1_conf": event["top_prob"],
    }
    return scores 

@tool("log_event", return_direct=False)
def log_event(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        persist the event + scores into a CSV file.
    """
    with open("events.csv", "a", newline="") as f:
        fieldnames = list(event.keys()) + list(scores.keys()) + ["ts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only if file is empty
        if f.tell() == 0:
            writer.writeheader()

        row = {**event, **scores, "ts": time.time()}
        writer.writerow(row)
    return "logged"

@tool("raise_alert", return_direct=False)
def raise_alert(event: Dict[str, Any], scores: Dict[str, Any]) -> str:
    """
        send an alert with a short rationale.
    """
    return f"ALERT: {event['top_label']} (p={event['top_prob']:.2f}) ent={event['entropy']:.2f} knn_dist={scores['knn_dist']:.2f}"


########## RAG (policy + incident) #######################################################
##########################################################################################


def retrieve_context(policies_store, incidents_store, event: Dict[str, Union[str, float]], scores: Dict[str, float]):
    """
        retrieve context from the knowledge base.
    """

    q = f"entropy={scores['ent']:.4f} knn_dist={scores['knn_dist']:.4f} mahalanobis={scores['mahalanobis']:.4f} top1_conf={scores['top1_conf']:.4f}"
    pol = "\n".join([d.page_content for d in policies_store.similarity_search(q, k=3)])
    inc = "\n".join([d.page_content for d in incidents_store.similarity_search(q, k=3)])
    return f"[POLICIES]\n{pol}\n\n[SIMILAR INCIDENTS]\n{inc}", q


########## Anomaly detection agent #######################################################
##########################################################################################

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a surveillance policy agent. You receive a perception event JSON, anomaly scores, "
     "and retrieved policies/incidents. Decide ONE action: LOG or ALERT."
     "Raise an ALERT if the observed action is abnormal; Log the event if the observed action is normal."
     "Return JSON with keys: action, rationale."),
    ("human", "Event:\n{event}\n\nScores:\n{scores}\n\nContext:\n{context}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
policy_chain = POLICY_PROMPT | llm | StrOutputParser()

def decide(policies_store, incidents_store, event: Dict[str, Any], scores: Dict[str, Any], gt_label: str) -> Dict[str, Any]:
    """
    make a decision based on the event, scores, and context.
    """
    ctx, q = retrieve_context(policies_store, incidents_store, event, scores)
    out = policy_chain.invoke({"event": event, "scores": scores, "context": ctx})
    # Remove Markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", out.strip(), flags=re.MULTILINE)

    decision = json.loads(cleaned)
    # insert the context into the knowledge base
    incident = f"[statistics]:{q} - [decision]:{decision['action']} | [ground_truth]:{gt_label}"
    incidents_store.add_texts([incident], metadatas=[{"kind": "incident_context"}])
    return decision

def process_window(policies_store, incidents_store, knn: DistanceScorer, model: CascadeFormerWrapper, skel_window: List[List[List[float]]], gt_label: str) -> Dict[str, Any]:
    # 1) Perceive
    event = perceive_window.invoke({"model": model, "skel_window": skel_window})

    # 2) Score anomaly
    scores = score_anomaly.invoke({"knn": knn, "event": event})

    # 3) Decide (your normal function)
    decision = decide(policies_store, incidents_store, event, scores, gt_label)

    # 4) Act/log via tools
    if decision["action"] == "ALERT":
        msg = raise_alert.invoke({"event": event, "scores": scores})
    else:
        msg = log_event.invoke({"event": event, "scores": scores})

    return {"event": event, "scores": scores, "decision": decision, "result": msg}


def prepare_one_sample(json_path: str, shuffle: bool, train: bool):
    split = 'train' if train else 'test'
    dataset = Feeder(
        data_path=DATA_PATH,
        split=split,
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
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)
    skeletons, labels, _ = next(iter(loader))  # (1,C,T,V,M)
    B, C, T, V, M = skeletons.shape

    sequences = skeletons.permute(0, 2, 3, 1, 4)
    # pick most active person
    motion = sequences.abs().sum(dim=(1, 2, 3))
    main_person_idx = motion.argmax(dim=-1)
    idx = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
    sequences = torch.gather(sequences, dim=4, index=idx).squeeze(-1)  # (1,T,V,C)
    window = sequences[0].cpu().numpy().astype(float)                  # (T,V,C)

    with open(json_path, "w") as f:
        json.dump(window.tolist(), f)
    return json_path, int(labels), window

def _get_limits(window, margin_ratio=0.1, use_3d=False):
    """Compute plot limits with a small margin from data."""
    if use_3d and window.shape[-1] >= 3:
        mins = window.min(axis=(0, 1))
        maxs = window.max(axis=(0, 1))
        span = np.maximum(maxs - mins, 1e-6)
        mins -= span * margin_ratio
        maxs += span * margin_ratio
        return (mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2])
    else:
        xy = window[..., :2]
        mins = xy.min(axis=(0, 1))
        maxs = xy.max(axis=(0, 1))
        span = np.maximum(maxs - mins, 1e-6)
        mins -= span * margin_ratio
        maxs += span * margin_ratio
        return (mins[0], maxs[0]), (mins[1], maxs[1])

def make_skeleton_video(
    window: np.ndarray,
    out_path: str,
    fps: int = 12,
    edges=NTU25_EDGES,
    title: str = "Skeleton Demo",
    use_3d: bool = True
):
    """
    Render a skeleton-only video from `window` of shape (T, 25, C).
    If use_3d=False, renders 2D using (x,y). If True and C>=3, renders 3D.
    """
    T, V, C = window.shape
    assert V == 25, f"Expected 25 joints; got {V}"
    assert C >= 2, "Need at least 2 channels (x,y)."

    # Matplotlib setup
    fig = plt.figure(figsize=(5, 5), dpi=200)
    if use_3d and C >= 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection='3d')
        (xlim, ylim, zlim) = _get_limits(window, use_3d=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    else:
        ax = fig.add_subplot(111)
        (xlim, ylim) = _get_limits(window, use_3d=False)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.invert_yaxis()

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if use_3d and C >= 3:
        ax.set_zticklabels([])

    ax.set_title(title)
    ax.set_aspect("equal")

    # Initialize artists: joints + bones
    if use_3d and C >= 3:
        joint_scatter = ax.scatter([], [], [], s=10)
        bone_lines = [ax.plot([], [], [], linewidth=1.5)[0] for _ in edges]
    else:
        joint_scatter = ax.scatter([], [], s=10)
        bone_lines = [ax.plot([], [], linewidth=1.5)[0] for _ in edges]

    def init():
        if use_3d and C >= 3:
            joint_scatter._offsets3d = ([], [], [])
            for ln in bone_lines:
                ln.set_data([], [])
                ln.set_3d_properties([])
        else:
            joint_scatter.set_offsets(np.empty((0, 2)))
            for ln in bone_lines:
                ln.set_data([], [])
        return [joint_scatter, *bone_lines]

    def update(t):
        pts = window[t]
        if use_3d and C >= 3:
            xs, ys, zs = pts[:, 0], pts[:, 2], pts[:, 1]  # swap y <-> z
            joint_scatter._offsets3d = (xs, ys, zs)
            for ln, (i, j) in zip(bone_lines, edges):
                ln.set_data([xs[i], xs[j]], [ys[i], ys[j]])
                ln.set_3d_properties([zs[i], zs[j]])
        else:
            xs, ys = pts[:, 0], pts[:, 1]
            joint_scatter.set_offsets(np.stack([xs, ys], axis=1))
            for ln, (i, j) in zip(bone_lines, edges):
                ln.set_data([xs[i], xs[j]], [ys[i], ys[j]])
        return [joint_scatter, *bone_lines]

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        blit=not (use_3d and C >= 3), interval=1000/fps
    )

    try:
        if animation.writers.is_available("ffmpeg"):
            ani.save(out_path, writer="ffmpeg", fps=fps, dpi=200, bitrate=2000)
            print(f"Saved {out_path}")
        elif animation.writers.is_available("pillow"):
            gif_path = out_path.replace(".mp4", ".gif")
            ani.save(gif_path, writer="pillow", fps=fps)
            print(f"ffmpeg not found; saved GIF instead: {gif_path}")
        else:
            print("No suitable writer found. Install ffmpeg or pillow to save the animation.")
    finally:
        plt.close(fig)


def run_on_single_json(policies_store, incidents_store, knn: DistanceScorer, model: CascadeFormerWrapper, json_path: str, gt_label: str):
    """
    Load one skeleton window from a JSON file and run the agent once.
    JSON format: nested list shaped like (T, J, C).
    """
    with open(json_path, "r") as f:
        skel_window = json.load(f)

    out = process_window(policies_store, incidents_store, knn, model, skel_window, gt_label)

    print("=== 🔥Demo Run🔥 ===", flush=True)
    print("🔥Decision:", json.dumps(out["decision"]["action"], indent=2), flush=True)
    print("🔥Rationale:", json.dumps(out["decision"]["rationale"], indent=2), flush=True)


def inference_demo(
        policies_store,
        incidents_store,
        model: CascadeFormerWrapper, 
        knn: DistanceScorer,
        json_path="demo_window.json",
        video_path="demo_window.mp4"
    ):
    # prepare one sample json
    _, _, window = prepare_one_sample(json_path, shuffle=True, train=False)

    # visualize the skeleton window
    make_skeleton_video(window, out_path=video_path, fps=12, use_3d=True)

    # run the agent on this json
    run_on_single_json(policies_store, incidents_store, knn, model, json_path)


def train_one_sample(
        policies_store,
        incidents_store,
        model: CascadeFormerWrapper, 
        knn: DistanceScorer,
        json_path="demo_window.json"
    ):
    # prepare one sample json
    _, label_id, _ = prepare_one_sample(json_path, shuffle=True, train=True)

    if is_abnormal_label(label_id):
        gt_label = "abnormal"
    else:
        gt_label = "normal"

    # run the agent on this json
    run_on_single_json(policies_store, incidents_store, knn, model, json_path, gt_label)


def print_incident_db(incidents_store):
    print("\n=== Incidents in the knowledge base ===", flush=True)
    for _, doc in incidents_store.docstore._dict.items():
        print(doc.page_content, flush=True)
        print("-------------------", flush=True)

def write_incident_db(incidents_store):
    with open("incidents_db.kb", "w") as f:
        for _, doc in incidents_store.docstore._dict.items():
            f.write(doc.page_content + "\n")
            f.write("-------------------\n")


def print_policy_db(policies_store):
    print("\n=== Policies in the knowledge base ===", flush=True)
    for _, doc in policies_store.docstore._dict.items():
        print(doc.page_content, flush=True)
        print("-------------------", flush=True)

def write_policy_db(policies_store):
    with open("policies_db.kb", "w") as f:
        for _, doc in policies_store.docstore._dict.items():
            f.write(doc.page_content + "\n")
            f.write("-------------------\n")

##########################################################################################
############## below is for RL-based policy optimization #################################

WEIGHTS = {"tp": +5, "fp": -3, "fn": -10, "tn": 0}

def compute_reward(gt_label, decision):
    """
        compute reward based on ground truth label and agent decision.
    """
    if gt_label == "abnormal" and decision == "ALERT":
        return WEIGHTS["tp"]
    elif gt_label == "normal" and decision == "ALERT":
        return WEIGHTS["fp"]
    elif gt_label == "abnormal" and decision == "LOG":
        return WEIGHTS["fn"]
    else:
        return WEIGHTS["tn"]


def train_a_reward_model(incidents_df: pd.DataFrame) -> RandomForestRegressor:
    """
    Train R(s,a) using your hand-crafted rewards as targets.
    Returns the fitted RandomForestRegressor.
    """
    # state features
    X = incidents_df[["entropy", "knn_dist", "mahalanobis", "top1_conf"]].values
    # action as 0/1
    A = (incidents_df["decision"].astype(str).str.upper() == "ALERT").astype(int).values.reshape(-1, 1)

    # label: numeric reward
    incidents_df = incidents_df.copy()
    incidents_df["reward"] = incidents_df.apply(
        lambda r: compute_reward(str(r.gt_label), str(r.decision)), axis=1
    )
    R = incidents_df["reward"].values

    # X_aug = [state, action]
    X_aug = np.concatenate([X, A], axis=1)
    r_model = RandomForestRegressor(max_depth=4, n_estimators=200, random_state=0)
    r_model.fit(X_aug, R)
    return r_model

@dataclass
class PolicyParams:
    # hard threshold rule
    max_entropy: float
    min_knn: float
    min_maha: float
    min_low_conf: float  # triggers when (1 - top1_conf) >= min_low_conf
    # optional linear score gate
    use_linear: bool = True
    w_entropy: float = 0.0
    w_knn: float = 1.0
    w_maha: float = 0.5
    w_lowconf: float = 0.5
    score_threshold: float = 0.8

def policy_decide(state, p: PolicyParams) -> str:
    """
    state: np.ndarray shape (4,) in order [entropy, knn_dist, mahalanobis, top1_conf]
    returns 'ALERT' or 'LOG'
    """
    ent, knn, maha, top1 = state
    lowconf = 1.0 - top1

    # Rule 1: any hard red-flag?
    if (ent >= p.max_entropy) or (knn >= p.min_knn) or (maha >= p.min_maha) or (lowconf >= p.min_low_conf):
        return "ALERT"

    # Rule 2: linear score gate (optional)
    if p.use_linear:
        score = (p.w_entropy*ent + p.w_knn*knn + p.w_maha*maha + p.w_lowconf*lowconf)
        if score >= p.score_threshold:
            return "ALERT"
    
    return "LOG"


def expected_return_of_policy(r_model: RandomForestRegressor, incidents_df: pd.DataFrame, p: PolicyParams) -> float:
    """
    Uses the learned R(s,a) to compute mean reward for policy p over a static set of states.
    """
    X = incidents_df[["entropy", "knn_dist", "mahalanobis", "top1_conf"]].values
    actions = []
    for s in X:
        a = policy_decide(s, p)
        actions.append(1 if a == "ALERT" else 0)
    A = np.array(actions, dtype=np.int32).reshape(-1, 1)
    X_aug = np.concatenate([X, A], axis=1)
    rewards = r_model.predict(X_aug)
    return float(np.mean(rewards))

def quantile_grid(values, qs):
    """
    compute quantiles of values at quantile levels qs.
    """
    qs = np.clip(np.asarray(qs), 0, 1)
    return np.quantile(values, qs)


def random_policy_search(r_model, incidents_df: pd.DataFrame, n_trials: int = 200, rng: int = 0) -> PolicyParams:
    """
    Simple hybrid search:
      - draws sensible ranges from feature quantiles
      - random samples + a few structured grid values
    """
    rs = np.random.RandomState(rng)
    X = incidents_df[["entropy", "knn_dist", "mahalanobis", "top1_conf"]].values
    ent, knn, maha, conf = X[:,0], X[:,1], X[:,2], X[:,3]
    lowconf = 1 - conf

    ent_q  = quantile_grid(ent,  [0.5, 0.7, 0.8, 0.9, 0.95])
    knn_q  = quantile_grid(knn,  [0.5, 0.7, 0.8, 0.9, 0.95])
    maha_q = quantile_grid(maha, [0.5, 0.7, 0.8, 0.9, 0.95])
    lc_q   = quantile_grid(lowconf, [0.5, 0.7, 0.8, 0.9, 0.95])

    best, best_ret = None, -1e9
   
    # random search before grid search
    grid = [(e, k, m, lc) for e in ent_q for k in knn_q for m in maha_q for lc in lc_q]
    rs.shuffle(grid)
    candidates = grid[: max(50, n_trials // 4)]
    for _ in range(n_trials - len(candidates)):
        e = rs.uniform(ent_q[0], ent_q[-1])
        k = rs.uniform(knn_q[0], knn_q[-1])
        m = rs.uniform(maha_q[0], maha_q[-1])
        lc = rs.uniform(lc_q[0],  lc_q[-1])
        candidates.append((e,k,m,lc))

    # grid search for the best policy parameters
    for (e,k,m,lc) in candidates:
        p = PolicyParams(
            max_entropy=float(e),
            min_knn=float(k),
            min_maha=float(m),
            min_low_conf=float(lc),
            use_linear=True,
            w_entropy=rs.uniform(0.0, 1.0),
            w_knn=rs.uniform(0.0, 1.0),
            w_maha=rs.uniform(0.0, 1.0),
            w_lowconf=rs.uniform(0.0, 1.0),
            score_threshold=rs.uniform(0.1, 2.0)
        )
        ret = expected_return_of_policy(r_model, incidents_df, p)
        if ret > best_ret:
            best, best_ret = p, ret

    return best


def decide_with_rl_policy(state_features: np.ndarray, learned_params: PolicyParams) -> Dict[str, Any]:
    """
    Deterministic decision using the learned policy parameters.
    """
    action = policy_decide(state_features, learned_params)
    return {
        "action": action,
        "rationale": f"RL policy thresholds {learned_params}"
    }

def extract_state_features(event: Dict[str,Any], scores: Dict[str,Any]) -> np.ndarray:
    """
    Aligns with training features: [entropy, knn_dist, mahalanobis, top1_conf]
    """
    ent = float(event["entropy"])
    knn = float(scores["knn_dist"])
    maha = float(scores["mahalanobis"])
    top1 = float(scores["top1_conf"])
    return np.array([ent, knn, maha, top1], dtype=float)


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
            windows = _select_main_person_batch(skeletons)

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

    # 2) Search best policy params under learned R(s,a)
    best_params = random_policy_search(r_model, incidents_df, n_trials=300, rng=42)
    print("\n=== RL-based Policy Optimization Result ===", flush=True)
    print("[RL] Best params:", best_params)
    print("\n===========================================", flush=True)

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

    # ==================================================
    # Build incidents_df directly (no caching)
    # ==================================================

    _ST_RE = re.compile(
        r"\[statistics\]:\s*"
        r"entropy=(?P<entropy>[+-]?\d+(?:\.\d+)?)\s+"
        r"knn_dist=(?P<knn_dist>[+-]?\d+(?:\.\d+)?)\s+"
        r"mahalanobis=(?P<mahalanobis>[+-]?\d+(?:\.\d+)?)\s+"
        r"top1_conf=(?P<top1_conf>[+-]?\d+(?:\.\d+)?)\s*-\s*"
        r"\[decision\]:(?P<decision>ALERT|LOG)\s*$"
    )

    def _parse_incidents_kb_file(path: str = "incidents_db.kb") -> pd.DataFrame:
        p = Path(path)
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("DUMMY") or set(s) == {"-"}:
                continue
            m = _ST_RE.search(s)
            d = m.groupdict()
            rows.append({
                "entropy": float(d["entropy"]),
                "knn_dist": float(d["knn_dist"]),
                "mahalanobis": float(d["mahalanobis"]),
                "top1_conf": float(d["top1_conf"]),
                "decision": d["decision"].upper(),
                "gt_label": "unknown",
            })
        return pd.DataFrame(rows)

    incidents_df = _parse_incidents_kb_file("incidents_db.kb")

    # Optionally merge FAISS metadata if available
    rows = []
    for _, doc in getattr(incidents_store.docstore, "_dict", {}).items():
        md = getattr(doc, "metadata", {}) or {}
        if str(md.get("kind","")).lower() not in ("incident","incident_context"):
            continue
        try:
            rows.append({
                "entropy": float(md.get("entropy", 0.0)),
                "knn_dist": float(md.get("knn_dist", 0.0)),
                "mahalanobis": float(md.get("mahalanobis", 0.0)),
                "top1_conf": float(md.get("top1_conf", md.get("top_prob", 0.0))),
                "decision": str(md.get("decision","LOG")).upper(),
                "gt_label": str(md.get("gt_label","unknown")).lower(),
            })
        except Exception:
            pass
    if rows:
        df_vs = pd.DataFrame(rows)
        incidents_df = pd.concat([incidents_df, df_vs], ignore_index=True)

    # --- Basic cleanup ---
    for c in ["entropy","knn_dist","mahalanobis","top1_conf"]:
        incidents_df[c] = pd.to_numeric(incidents_df[c], errors="coerce")
    incidents_df["decision"] = incidents_df["decision"].astype(str).str.upper()
    incidents_df["gt_label"] = incidents_df["gt_label"].astype(str).str.lower()
    mask_all_nan = incidents_df[["entropy","knn_dist","mahalanobis","top1_conf"]].isna().all(axis=1)
    incidents_df = incidents_df.loc[~mask_all_nan].reset_index(drop=True)
    for c in ["entropy","knn_dist","mahalanobis","top1_conf"]:
        if incidents_df[c].isna().any():
            incidents_df[c] = incidents_df[c].fillna(incidents_df[c].median())

    print(f"[incidents_df] {len(incidents_df)} rows | columns: {list(incidents_df.columns)}")

    # ==================================================
    # Run RL optimization
    # ==================================================
    rl_policy_optimization(
        incidents_df,
        policies_store, incidents_store,
        knn, model,
        device="cuda"
    )



##########################################################################################
############## below is for offline evaluation ###########################################
##########################################################################################

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


def is_abnormal_label(label_id: int) -> bool:
    return classify_labels[label_id] in set(abnormal_action_labels)

def _select_main_person_batch(skeletons: torch.Tensor) -> torch.Tensor:
    """
    (B,C,T,V,M) -> (B,T,V,C) using your 'most active person' heuristic.
    """
    B, C, T, V, M = skeletons.shape
    sequences = skeletons.permute(0, 2, 3, 1, 4)              # (B,T,V,C,M)
    motion = sequences.abs().sum(dim=(1, 2, 3))               # (B,M)
    main_person_idx = motion.argmax(dim=-1)                   # (B,)
    idx = main_person_idx.view(B, 1, 1, 1, 1).expand(-1, T, V, C, 1)
    sequences = torch.gather(sequences, dim=4, index=idx).squeeze(-1)  # (B,T,V,C)
    return sequences


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
            windows = _select_main_person_batch(skeletons)   # (B,T,V,C)

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
            windows = _select_main_person_batch(skeletons)   # (B,T,V,C)

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

def agent_training_and_demo():
    INFERENCE_ONLY = False
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

if __name__ == "__main__":

    # mode 1: running agent training (optional) + a random inference demo
    agent_training_and_demo()


    # mode 2: reinforcement-learning based policy optimization
    #agent_rl_policy_optimization()


