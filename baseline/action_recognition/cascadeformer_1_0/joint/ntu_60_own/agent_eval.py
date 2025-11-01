import os
import joblib
import matplotlib
matplotlib.use("Agg")
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from agent_components.perceiver import CascadeFormerWrapper
from agent_components.statistics import DistanceScorer
from agent_components.rag import print_incident_db, print_policy_db
from agent_components.eval import evaluate_full_test_split_with_agent, evaluate_random_batches_with_agent, evaluate_full_test_split_policy_only
from agent_components.reinforcement import PolicyParams

def main():
    MODE = "random" # 'full' or 'random' or 'policy_only'
    POLICY = "RL" # 'RL' 'classification'



    model = CascadeFormerWrapper(device="cuda")
    # ---- Load or build KNN scorer ----
    if not os.path.exists("trained_knn.pkl"):
        knn = DistanceScorer(model=model)
        joblib.dump(knn, "trained_knn.pkl")
    knn_scorer = joblib.load("trained_knn.pkl")

    # ---- Load vector stores ----
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    incidents_store = FAISS.load_local("vectorstores/incidents", emb, allow_dangerous_deserialization=True)
    
    if POLICY == "classification":
        policies_store  = FAISS.load_local("vectorstores/initial_policies",  emb, allow_dangerous_deserialization=True)
    elif POLICY == "RL":
        policies_store  = FAISS.load_local("vectorstores/rl_learned_policies",  emb, allow_dangerous_deserialization=True)
    else:
        raise ValueError(f"Unknown POLICY: {POLICY}")
    
    print_incident_db(incidents_store)
    print_policy_db(policies_store)

    if MODE == "full":
        evaluate_full_test_split_with_agent(
            policies_store,
            incidents_store,
            knn_scorer,
            model,
            batch_size=16,
            device="cuda"
        )
    elif MODE == "random":
        NUM_BATCHES = 20
        evaluate_random_batches_with_agent(
            policies_store,
            incidents_store,
            knn_scorer,
            model,
            num_batches=NUM_BATCHES,
            batch_size=16,
            device="cuda"
        )
    elif MODE == "policy_only":

        # CascadeFormer 2
        PARAMS = PolicyParams(
            max_entropy = 0.3821,
            min_knn = 0.1736,
            min_maha = 19.9780,
            min_low_conf = 1 - 0.0471
        )
        evaluate_full_test_split_policy_only(
            knn_scorer,
            model,
            PARAMS,
            batch_size=16,
            device="cuda"
        )


if __name__ == "__main__":
    main()
