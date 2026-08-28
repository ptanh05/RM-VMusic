# RM-VMusic Phase 9: Formal Methodological & Mathematical Specification of UAD-Fusion

This document specifies the exact mathematical formulations, loss functions, and architectural layers of the **Uncertainty-Aware Dynamic Multimodal Fusion (UAD-Fusion)** network.

---

## 1. Modality-Specific Encoders

Let a music instance be represented by the triplet of physical modalities $(x_{\text{lyrics}}, x_{\text{cover}}, x_{\text{audio}})$ and binary availability masks $(m_{\text{lyrics}}, m_{\text{cover}}, m_{\text{audio}}) \in \{0, 1\}^3$.

Each modality is mapped to a shared projection dimension $d = 256$:
$$h_l = \text{Encoder}_l(x_{\text{lyrics}}) \odot m_{\text{lyrics}} \in \mathbb{R}^{256}$$
$$h_c = \text{Encoder}_c(x_{\text{cover}}) \odot m_{\text{cover}} \in \mathbb{R}^{256}$$
$$h_a = \text{Encoder}_a(x_{\text{audio}}) \odot m_{\text{audio}} \in \mathbb{R}^{256}$$

---

## 2. Dynamic Uncertainty & Reliability Estimation

For each active modality $m \in \{l, c, a\}$, an MLP uncertainty estimator predicts an unconstrained log-variance score $s_m$:
$$u_m = \text{Softplus}\left(W_{u,m} h_m + b_{u,m}\right) + (1.0 - m_m) \cdot 10.0$$

The dynamic fusion weights $w_m$ are computed via normalized inverse-uncertainty softmax:
$$w_m = \frac{\exp(-u_m)}{\sum_{k \in \{l, c, a\}} \exp(-u_k)}, \quad \sum_{m} w_m = 1.0$$

---

## 3. Dynamic Representation Fusion & Classification

The uncertainty-weighted representations are concatenated and projected:
$$h_{\text{fused}} = \text{LayerNorm}\left(\text{LeakyReLU}\left(W_f \left[ w_l h_l \,\|\, w_c h_c \,\|\, w_a h_a \right] + b_f\right)\right) \in \mathbb{R}^{512}$$
$$\hat{y} = \text{Softmax}\left(W_{\text{cls}} h_{\text{fused}} + b_{\text{cls}}\right) \in \mathbb{R}^{12}$$

---

## 4. Supervised Contrastive Multimodal Loss

To enforce intra-genre compactness and inter-genre separability under missing modalities:
$$\mathcal{L}_{\text{supcon}} = \sum_{i=1}^{B} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp\left(\frac{z_i \cdot z_p}{\tau}\right)}{\sum_{a \in A(i)} \exp\left(\frac{z_i \cdot z_a}{\tau}\right)}$$
where $z_i = \frac{h_{\text{fused}, i}}{\|h_{\text{fused}, i}\|_2}$ and $\tau = 0.10$.

---

## 5. Total Training Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\hat{y}, y; w_{\text{class}}) + \lambda_{\text{supcon}} \mathcal{L}_{\text{supcon}}$$
with balanced class weighting $w_{\text{class}} = \frac{N}{12 \cdot N_c}$ and $\lambda_{\text{supcon}} = 0.15$.
