# 🌊 CascadeFormer: A Family of Two-stage Cascading Transformers for Skeleton-based Human Action Recognition

## News/Updates

- [September 16, 2025] paper under review at ICLR 2026!
- [August 31, 2025] paper available on [arXiv](https://arxiv.org/abs/2509.00692)!
- [July 19, 2025] model checkpoints are publicly available on [HuggingFace](https://huggingface.co/YusenPeng/CascadeFormerCheckpoints) for further analysis/application!

## CascadeFormer

![alt text](docs/CascadeFormer_pretrain.png)

Overview of the masked pretraining component in CascadeFormer. A fixed percentage of joints are randomly masked across all frames in each video. The partially masked skeleton sequence is passed through a feature extraction module to produce frame-level embeddings, which are then input into a temporal transformer (T1). A lightweight linear decoder is applied to reconstruct the masked joints, and the model is optimized using mean squared error over the masked positions. This stage enables the model to learn generalizable spatiotemporal representations prior to supervised finetuning.

![alt text](docs/CascadeFormer_finetune.png)

Overview of the cascading finetuning component in CascadeFormer. The frame embeddings produced by the pre-
trained temporal transformer backbone (T1) are passed into a task-specific transformer (T2) for hierarchical refinement. The output of T2 is fused with the original embeddings via a cross-attention module. The resulting fused representations are aggregated through frame-level average pooling and passed to a lightweight classification head. The entire model—including T1, T2, and the classification head—is optimized using cross-entropy loss on action labels during finetuning.

## Evaluation

![alt text](docs/eval_results.png)

Overall accuracy evaluation results of CascadeFormer variants on three datasets. CascadeFormer 1.0 consistently
achieves the highest accuracy on Penn Action and both NTU60 splits, while 1.1 excels on N-UCLA. All checkpoints are open-
sourced for reproducibility.

## Citation

Please cite our work if you find it useful/helpful:

```bibtex
@misc{peng2025cascadeformerfamilytwostagecascading,
      title={CascadeFormer: A Family of Two-stage Cascading Transformers for Skeleton-based Human Action Recognition}, 
      author={Yusen Peng and Alper Yilmaz},
      year={2025},
      eprint={2509.00692},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.00692}, 
}
```

## Contacts

If you have any questions or suggestions, feel free to contact:

- Yusen Peng (peng.1007@osu.edu)
- Alper Yilmaz (yilmaz.15@osu.edu)

Or describe it in Issues.


## 🔥 Ongoing follow-up Work: CascadeFormer-Agent for anomaly detection

### CascadeFormer-Agent

![alt text](/CascadeFormer-AD-Agent.png)

### Commands

```bash
# export the API key first
export OPENAI_API_KEY=<API KEY GOES HERE>
# log incidents with statistics (optional) + inference demo
CUDA_VISIBLE_DEVICES=0 taskset -c 20-30 python baseline/action_recognition/cascadeformer_1_0/joint/ntu_60_own/agent_demo.py
# reinforcement-learning
CUDA_VISIBLE_DEVICES=0 taskset -c 20-30 python baseline/action_recognition/cascadeformer_1_0/joint/ntu_60_own/agent_RL.py
```

### Logging incidents during agent training

```csharp
=== Incidents in the knowledge base ===
DUMMY INCIDENT ENTRY; DO NOT USE.
-------------------
[statistics]:entropy=0.3807 knn_dist=0.1431 mahalanobis=18.3442 top1_conf=0.9530 - [decision]:LOG | [ground_truth]:normal
-------------------
[statistics]:entropy=0.3434 knn_dist=1.1534 mahalanobis=89.0360 top1_conf=0.9583 - [decision]:ALERT | [ground_truth]:abnormal
-------------------
[statistics]:entropy=0.3512 knn_dist=0.3321 mahalanobis=34.6292 top1_conf=0.9570 - [decision]:LOG | [ground_truth]:normal
-------------------
[statistics]:entropy=0.3273 knn_dist=1.1033 mahalanobis=80.0809 top1_conf=0.9603 - [decision]:ALERT | [ground_truth]:abnormal
-------------------
[statistics]:entropy=0.3980 knn_dist=1.2050 mahalanobis=85.4945 top1_conf=0.9506 - [decision]:ALERT | [ground_truth]:abnormal
-------------------
[statistics]:entropy=0.3801 knn_dist=0.0901 mahalanobis=12.8214 top1_conf=0.9531 - [decision]:LOG | [ground_truth]:normal
-------------------
[statistics]:entropy=0.3672 knn_dist=0.1526 mahalanobis=21.0568 top1_conf=0.9549 - [decision]:LOG | [ground_truth]:normal
-------------------
[statistics]:entropy=0.2880 knn_dist=0.3832 mahalanobis=43.8138 top1_conf=0.9657 - [decision]:ALERT | [ground_truth]:normal
-------------------
[statistics]:entropy=0.3586 knn_dist=0.2176 mahalanobis=23.3810 top1_conf=0.9561 - [decision]:LOG | [ground_truth]:normal
-------------------
[statistics]:entropy=0.3880 knn_dist=0.0858 mahalanobis=12.3786 top1_conf=0.9520 - [decision]:LOG | [ground_truth]:normal
-------------------
```

### RL-assisted grid search for policy thresholds

```csharp
=== RL-based Policy Optimization Result ===
[RL] Best params: PolicyParams(max_entropy=0.389, min_knn=0.5992, min_maha=54.6939, min_low_conf=0.0481)
[RL] Highest reward: 1.154
===========================================
```

### test-set evaluation (preliminary)

```csharp
=== Offline Evaluation (RL Policy over RANDOM TEST Batches) ===
Samples   : 200
Accuracy  : 0.4900
Precision : 0.3108
Recall    : 1.0000
F1-score  : 0.4742
```
