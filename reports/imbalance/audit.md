# RM-VMusic: Long-Tail Class Imbalance & Batch Sampling Forensic Audit
**Evaluation Date:** 2026-08-28  
**Dataset Catalog:** Dataset V4 ($N = 8,559$ total, Train $N = 5,991$)  
**Taxonomy Space:** 12 Mutually Exclusive Genres

---

## 1. Class Distribution & Batch Sampling Dynamics (Batch Size = 64)

| Genre Class | Taxonomy Tier | Train $N$ (%) | Test $N$ | Inverse Class Weight ($w_c$) | Uniform Batch Exp. (Zero-Prob) | Weighted Sampler Exp. (Zero-Prob) |
|---|---|---|---|---|---|---|
| `POP_BALLAD` | **HEAD (Majority)** | 2,150 (35.89%) | 461 | **0.232** | 22.97 (0.00%) | 5.33 (0.38%) |
| `FOLK_TRADITIONAL` | **HEAD (Majority)** | 1,890 (31.55%) | 405 | **0.264** | 20.19 (0.00%) | 5.33 (0.38%) |
| `BOLERO_TRUTINH` | **HEAD (Majority)** | 570 (9.51%) | 122 | **0.876** | 6.09 (0.17%) | 5.33 (0.38%) |
| `INSTRUMENTAL` | **MEDIUM (Body)** | 307 (5.12%) | 66 | **1.626** | 3.28 (3.45%) | 5.33 (0.38%) |
| `REVOLUTIONARY` | **MEDIUM (Body)** | 189 (3.15%) | 41 | **2.642** | 2.02 (12.85%) | 5.33 (0.38%) |
| `RAP_HIPHOP` | **MEDIUM (Body)** | 155 (2.59%) | 33 | **3.221** | 1.66 (18.68%) | 5.33 (0.38%) |
| `ROCK` | **TAIL (Minority)** | 152 (2.54%) | 32 | **3.285** | 1.62 (19.31%) | 5.33 (0.38%) |
| `RB_SOUL` | **TAIL (Minority)** | 148 (2.47%) | 32 | **3.373** | 1.58 (20.17%) | 5.33 (0.38%) |
| `DANCE_EDM` | **TAIL (Minority)** | 137 (2.29%) | 29 | **3.644** | 1.46 (22.75%) | 5.33 (0.38%) |
| `CHILDREN` | **TAIL (Minority)** | 121 (2.02%) | 26 | **4.126** | 1.29 (27.09%) | 5.33 (0.38%) |
| `NHAC_TRINH` | **TAIL (Minority)** | 102 (1.7%) | 22 | **4.895** | 1.09 (33.32%) | 5.33 (0.38%) |
| `OTHER` | **TAIL (Minority)** | 70 (1.17%) | 15 | **7.132** | 0.75 (47.13%) | 5.33 (0.38%) |

---

## 2. Answers to Core Forensic Audit Questions

### Q1: Is class weighting currently used, and what is the exact formula?
- **Finding:** Class weighting is implemented in `src/training/trainer.py` as:
  $$w_c = \frac{N_{\text{train}}}{C \cdot (N_{c,\text{train}} + 1.0)} \times \frac{C}{\sum_k w_k}$$
- **Verification:** It is computed **strictly on `train_ds.labels`** (TRAIN ONLY). There is zero contamination from validation or test splits.

### Q2: Is WeightedRandomSampler currently applied?
- **Finding:** Currently, `train_loader` uses standard uniform sampling (`shuffle=True`). `WeightedRandomSampler` is not active in the master benchmark.
- **Val / Test Safety:** Validation and Test loaders use `shuffle=False` and uniform natural evaluation.

### Q3: Analysis of the Claim: *"WeightedRandomSampler ensures ~5 samples per class in each batch of 64"*
- **Mathematical Reality:**
  1. `WeightedRandomSampler` enforces an **expected value** of $\mathbb{E}[N_c] = \frac{64}{12} \approx 5.33$ samples per class.
  2. Because sampling is **with replacement**, the distribution is a Multinomial draw $M(64, [1/12, \dots, 1/12])$.
  3. The standard deviation is $\sigma = \sqrt{64 \times \frac{1}{12} \times \frac{11}{12}} \approx 2.21$ samples.
  4. In any single batch, a class typically has between **2 and 9 samples**, with a **$0.40\%$ probability of having 0 samples**.
  5. Thus, it is an **expectation-based balancing mechanism**, not a deterministic partition.

### Q4: Risk of Over-Compensation (Double Penalty)
- **Warning:** Combining `WeightedRandomSampler` (which already boosts minority class sampling frequency by $\approx \frac{1}{N_c}$) with `WeightedCrossEntropyLoss` (which scales loss by $w_c \approx \frac{1}{N_c}$) results in an effective gradient scaling of $\mathcal{O}\left(\frac{1}{N_c^2}\right)$.
- **Consequence:** Minority classes with noisy labels or high ambiguity will dominate the loss gradients, causing training instability and degrading overall calibration (higher ECE).

---

## 3. Definition of Head, Medium, and Tail Groups
- **HEAD (Majority - Top 3):** `POP_BALLAD` ($2,150$ in train), `FOLK_TRADITIONAL` ($1,890$ in train), `BOLERO_TRUTINH` ($570$ in train). Accounts for **$76.95\%$** of training data.
- **MEDIUM (Body - Mid 3):** `INSTRUMENTAL` ($307$), `REVOLUTIONARY` ($189$), `RAP_HIPHOP` ($155$). Accounts for **$10.87\%$** of training data.
- **TAIL (Minority - Bottom 6):** `ROCK` ($152$), `RB_SOUL` ($148$), `DANCE_EDM` ($137$), `CHILDREN` ($121$), `NHAC_TRINH` ($102$), `OTHER` ($70$). Accounts for **$12.18\%$** of training data.
