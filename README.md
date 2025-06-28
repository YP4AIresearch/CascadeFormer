# 🌊 CascadeFormer: Two-stage Cascading Transformer for Human Action Recognition

## Case study for CascadeFormer 1.1 (with convolution):

### Tuning the model size won't help...

1. layers (fix hidden size = 512, heads = 8): 8 layers: 73.07%; 12 layers: 73.97%
2. hidden size (fix layers = 8, heads = 8): 512: 73.97%; 768: 73.48%; 1024: 73.61%

### Strong data augmentation won't help...

1. 100% data augmentation instead of just 10%?
2. 73.61% -> 73.33%...

### Multi-stream input?

pending - separate T1 encoders, fuse features after T1 [under discussion]

### SoTA of NTU

![alt text](docs/NTU_SoTA.png)

### Reference Training setup (SkateFormer, HyperFormer)

![alt text](docs/reference_training_setup.png)

## CascadeFormer 1.X series

![alt text](docs/CascadeFormer_1.png)

### leaderboard - CascadeFormer 1.0

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **94.66%** > 93.4% (HDM-BG) [google drive](https://drive.google.com/drive/folders/1Za50ZE9ZEKdEps_ZE-JIbatTpLuMW83k) |
| N-UCLA | 1,494 | 12 | 3D | 20 | **88.79%** < 98.3% (SkateFormer) |
| NTU/CS | 56,880 | 60 | 3D | 25 | **75.22%** << 92.6% (SkateFormer) - cross subject |
| NTU/CD | 56,880 | 60 | 3D | 25 | TBD |
| <tr><td colspan="6" align="center"> **Bone** data representation ablation study </td></tr> |
| Penn Action, subtraction-bone | 2,326 | 15 | 2D | 13 | **92.32%** ~ 93.4% (HDM-BG) |
| Penn Action, concatenation-bone | 2,326 | 15 | 2D | 13 | **93.16%** ~ 93.4% (HDM-BG) |
| Penn Action, parameterization-bone | 2,326 | 15 | 2D | 13 | **93.91%** > 93.4% (HDM-BG) |
| N-UCLA, subtraction-bone | 1,494 | 12 | 3D | 20 | **85.56%** < 98.3% (SkateFormer) |
| N-UCLA, concatenation-bone | 1,494 | 12 | 3D | 20 | **88.15%** < 98.3% (SkateFormer) |
| NTU/CS, subtraction-bone | 56,880 | 60 | 3D | 25 | **74.23%** << 92.6% (SkateFormer) - cross subject |
| NTU/CS, concatenation-bone | 56,880 | 60 | 3D | 25 | **73.81%** << 92.6% (SkateFormer) - cross subject |

### leaderboard - CascadeFormer 1.1

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **94.10%** > 93.4% (HDM-BG) [google drive](https://drive.google.com/drive/folders/1qbcT8DlhNyT3HgbM3j2aEQP2rSXoEJRS) |
| N-UCLA | 1,494 | 12 | 3D | 20 | **91.16%** < 98.3% (SkateFormer) [google drive](https://drive.google.com/drive/folders/1b0IuO_XY-Gwv4RjS6gF9gPG36uvGwhha); **90.52%** [google drive](https://drive.google.com/drive/folders/10v1zGGhziiRZdXO2mDU-db_keVmmeUNY) |
| NTU/CS | 56,880 | 60 | 3D | 25 | **74.10%** << 92.6% (SkateFormer) |
| NTU/CV | 56,880 | 60 | 3D | 25 | ??? << 92.6% (SkateFormer) |

### leaderboard - CascadeFormer 1.2

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **94.10%** > 93.4% (HDM-BG) [google drive](https://drive.google.com/drive/folders/1Jl7lIVcbqw6W2xzvf09nVRERXHIFrjXn); **94.01%** [google drive](https://drive.google.com/drive/folders/1jAlH7pf-zaHy7CVIF3MAmiZ5mMtDw2j-) |
| N-UCLA | 1,494 | 12 | 3D | 20 | **90.73%** < 98.3% (SkateFormer) [google drive](https://drive.google.com/drive/folders/1IPSW5pz_Sn0dfywP2RatlnlrfVzPJNvB) |
| NTU/CS | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |
| NTU/CV | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |

### leaderboard - CascadeFormer 1.3

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | 93.4% (HDM-BG) |
| N-UCLA | 1,494 | 12 | 3D | 20 | 98.3% (SkateFormer) |
| NTU/CS | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |
| NTU/CV | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |


## CascadeFormer 2.0 (interleaved spatial–temporal attention inspired by [IIP-Transformer](https://arxiv.org/abs/2110.13385) and [ST-TR](https://arxiv.org/abs/2012.06399))  

![alt text](docs/CascadeFormer_2.png)

### result leaderboard - CascadeFormer 2.0

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **92.32%** = 93.4% (HDM-BG) |
| N-UCLA | 1,494 | 12 | 3D | 20 | 98.3% (SkateFormer) |
| NTU/CS | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |
| NTU/CV | 56,880 | 60 | 3D | 25 | 92.6% (SkateFormer) |

corresponding model checkpoints:

1. Penn Action: **92.32%** [google drive](https://drive.google.com/drive/folders/1cYQMhedWKBm93L9RWSEAj2HYGhdlucKl) - for Penn Action at least, it's very sensitive to overfitting! (sometimes fail to converge too...)
2. N-UCLA: TBD
3. NTU/CS: TBD
4. NTU/CV: TBD
