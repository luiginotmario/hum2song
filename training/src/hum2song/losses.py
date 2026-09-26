"""Supervised contrastive loss, query-type cross-entropy, and batch confidence."""

import torch
import torch.nn.functional as F

MASK_LOGIT = -1.0e4


def info_nce_symmetric(
    query: torch.Tensor,
    song: torch.Tensor,
    song_ids: list[str],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE. Same-song off-diagonal pairs are masked, not used as negatives."""
    logits = query @ song.transpose(0, 1) / temperature
    logits = logits.masked_fill(_same_song_mask(song_ids, logits.device), MASK_LOGIT)
    targets = torch.arange(query.shape[0], device=query.device)
    loss_query = F.cross_entropy(logits, targets)
    loss_song = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (loss_query + loss_song)


def qtype_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for the hum / whistle / sing head."""
    return F.cross_entropy(logits, targets)


def confidence_loss(logits: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal-style push of the positive softmax probability toward 1."""
    probabilities = torch.softmax(logits, dim=-1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    positive = probabilities.gather(1, targets[:, None]).squeeze(1)
    return ((1.0 - positive) ** gamma).mean()


def batch_ece(logits: torch.Tensor, n_bins: int = 10) -> float:
    """Expected calibration error of the in-batch positive probability."""
    probabilities = torch.softmax(logits.detach(), dim=-1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    confidence = probabilities.gather(1, targets[:, None]).squeeze(1)
    prediction = probabilities.argmax(dim=-1)
    correct = (prediction == targets).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    total = 0.0
    count = logits.shape[0]
    for index in range(n_bins):
        low = edges[index]
        high = edges[index + 1]
        high_inclusive = index == n_bins - 1
        under_high = confidence <= high if high_inclusive else confidence < high
        chosen = (confidence >= low) & under_high
        chosen_count = int(chosen.sum().item())
        if chosen_count == 0:
            continue
        gap = abs(float(correct[chosen].mean()) - float(confidence[chosen].mean()))
        total += gap * chosen_count / count
    return total


def masked_logits(
    query: torch.Tensor,
    song: torch.Tensor,
    song_ids: list[str],
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Query-to-song logits with same-song false negatives removed. Used by L_conf."""
    logits = query @ song.transpose(0, 1) / temperature
    return logits.masked_fill(_same_song_mask(song_ids, logits.device), MASK_LOGIT)


def _same_song_mask(song_ids: list[str], device: torch.device) -> torch.Tensor:
    size = len(song_ids)
    mask = torch.zeros(size, size, dtype=torch.bool, device=device)
    groups: dict[str, list[int]] = {}
    for index, song_id in enumerate(song_ids):
        groups.setdefault(song_id, []).append(index)
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        for left in indexes:
            for right in indexes:
                if left != right:
                    mask[left, right] = True
    return mask
