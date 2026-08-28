"""
Loss functions for RM-VMusic training:
- Balanced Cross-Entropy Loss
- Supervised Contrastive Loss (SupCon)
- Modality Reliability Regularization
- Distribution Invariance Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.criterion(logits, targets)

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss over fused representations.
    Encourages representations of the same music genre to cluster tightly.
    """
    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D]
        # labels: [B]
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for same-class pairs
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # For numerical stability
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Compute log-sum-exp over all non-self elements
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class DistributionInvarianceLoss(nn.Module):
    """
    Penalizes disparity in fused representations across modality presence masks.
    """
    def __init__(self):
        super().__init__()

    def forward(self, fused_emb, modality_masks):
        # fused_emb: [B, D]
        # modality_masks: [B, 3]
        # Encourage similarity between complete and missing representations
        norm_emb = F.normalize(fused_emb, p=2, dim=-1)
        cos_sim = torch.matmul(norm_emb, norm_emb.T)
        
        mask_diff = torch.cdist(modality_masks.float(), modality_masks.float(), p=1)
        # Minimize representation distance when masks differ
        invariance_loss = ((1.0 - cos_sim) * (mask_diff > 0).float()).mean()
        return invariance_loss
