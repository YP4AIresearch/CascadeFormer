import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass

from .constants import WEIGHTS, SEARCH_BARS

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

def policy_decide(state, p: PolicyParams) -> str:
    """
    state: np.ndarray shape (4,) in order [entropy, knn_dist, mahalanobis, top1_conf]
    returns 'ALERT' or 'LOG'
    """
    ent, knn, maha, top1 = state
    # lowconf = 1.0 - top1

    # if (ent >= p.max_entropy) or (knn >= p.min_knn) or (maha >= p.min_maha) or (lowconf >= p.min_low_conf):
    #     return "ALERT"
    # else:
    #     return "LOG"
    if (knn >= p.min_knn) or (maha >= p.min_maha):
        return "ALERT"
    else:
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

def policy_search(r_model, incidents_df: pd.DataFrame) -> PolicyParams:
    X = incidents_df[["entropy", "knn_dist", "mahalanobis", "top1_conf"]].values
    ent, knn, maha, conf = X[:,0], X[:,1], X[:,2], X[:,3]
    lowconf = 1 - conf

    ent_q  = quantile_grid(ent, SEARCH_BARS)
    knn_q  = quantile_grid(knn, SEARCH_BARS)
    maha_q = quantile_grid(maha, SEARCH_BARS)
    lc_q   = quantile_grid(lowconf, SEARCH_BARS)

    best, best_ret = None, -1e9
   
    # search space
    candidates = [(e, k, m, lc) for e in ent_q for k in knn_q for m in maha_q for lc in lc_q]  # 625
    # grid search for the best policy parameters
    for (e,k,m,lc) in tqdm(candidates, desc="Searching policies"):
        p = PolicyParams(
            max_entropy=round(float(e), 4),
            min_knn=round(float(k), 4),
            min_maha=round(float(m), 4),
            min_low_conf=round(float(lc), 4)
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
