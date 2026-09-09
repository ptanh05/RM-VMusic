# UAD-Fusion official-v1 specification

For each modality m in {lyrics, cover, audio}, a two-layer encoder maps a finite feature vector to z_m in R^256. The binary mask is applied to the raw feature before encoding, so an unavailable input cannot affect the prediction through arbitrary cached values.

The learned uncertainty head produces

```text
u_m = Softplus(g_m(z_m))
r_m = exp(-u_m) * mask_m
alpha_m = exp(-u_m) / sum_j exp(-u_j), over available modalities only
z_fused = sum_m alpha_m * z_m
```

Unavailable modalities have exactly zero reliability and fusion weight. Computation uses masked softmax for numerical stability. When every mask is zero, all fusion weights and `z_fused` are zero; `has_evidence=False` is returned and the classifier bias yields finite logits. Such predictions are counted separately during evaluation.

During training, independent modality dropout is applied only to already available masks. If every available modality would be dropped, the original mask is restored. Evaluation does not apply model-side dropout.

The forward result exposes `uncertainty`, `reliability`, `fusion_weights`, `modality_weights`, `active_masks`, `has_evidence`, modality embeddings, the fused embedding and logits.

The scores are learned positive reliability scores. They have not been established as calibrated aleatoric variance estimates; “inverse variance” should not be claimed without a suitable probabilistic objective and validation.

The official objective is weighted cross-entropy plus supervised contrastive loss on evidence-bearing fused representations. The old unpaired distribution-invariance loss is disabled because it attracted representations from unrelated labels solely on the basis of different modality masks.

Architecture version 1 is stored in the model state and rejects legacy sigmoid-gate checkpoints. Old experimental results therefore cannot be attributed to this corrected implementation without retraining.
