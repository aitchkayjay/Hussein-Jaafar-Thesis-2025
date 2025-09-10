import torch
import torch.nn.functional as F
from typing import Optional

@torch.no_grad()
def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B,H,W] (long) -> [B,C,H,W] (float)
    return F.one_hot(labels.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()

def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    multiclass: bool = True,
    epsilon: float = 1.0,
    ignore_index: Optional[int] = None,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generalized Dice Loss.
    - logits: [B,C,H,W] (multiclass) oder [B,1,H,W] (binary)
    - target: [B,H,W]   (class IDs bei multiclass, 0..C-1), int64
    - multiclass=True -> Softmax über C; one-hot Targets
      multiclass=False -> Sigmoid, target muss 0/1 sein (IDs werden intern auf {0,1} gemappt)
    - ignore_index: z. B. 0 (void). Diese Pixel werden aus der Dice-Berechnung ausgeschlossen.
    - weight: optional pro-Klassen-Gewichte [C] (nur multiclass)

    Rückgabe: skalarer Loss (1 - mean Dice)
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be [B,C,H,W], got {tuple(logits.shape)}")
    b, c, h, w = logits.shape
    if target.shape != (b, h, w):
        raise ValueError(f"target must be [B,H,W], got {tuple(target.shape)}")

    if multiclass:
        # Softmax -> [B,C,H,W]
        prob = F.softmax(logits, dim=1)

        # Ignore-Maske vorbereiten
        if ignore_index is not None:
            valid = (target != ignore_index)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        # One-hot nur für gültige Pixel
        oh = _one_hot(target, num_classes=c)  # [B,C,H,W]

        # gültige Pixel maskieren
        valid = valid.unsqueeze(1)  # [B,1,H,W]
        prob = prob * valid
        oh   = oh   * valid

        # Per-Klasse Dice
        intersect = (prob * oh).sum(dim=(0, 2, 3))            # [C]
        denom     = prob.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))  # [C]
        dice_c    = (2.0 * intersect + epsilon) / (denom + epsilon)  # [C]

        if weight is not None:
            if weight.shape[0] != c:
                raise ValueError(f"weight must have shape [C]={c}, got {tuple(weight.shape)}")
            # Klassen, die vollständig ignoriert/leer sind, sollen Gewicht 0 erhalten
            valid_cls = (denom > 0)
            w = torch.zeros_like(dice_c)
            w[valid_cls] = weight[valid_cls]
            loss = 1.0 - (dice_c * w).sum() / (w.sum() + 1e-8)
        else:
            # Mittel über Klassen, die überhaupt vorkommen (denom>0)
            valid_cls = (denom > 0)
            if valid_cls.any():
                loss = 1.0 - dice_c[valid_cls].mean()
            else:
                loss = logits.new_tensor(0.0)  # nichts zu lernen
        return loss

    else:
        # Binary: Sigmoid + binärer Dice (Klasse = 1; alles andere = 0)
        prob = torch.sigmoid(logits)  # [B,1,H,W]
        if logits.size(1) != 1:
            raise ValueError("For binary dice_loss, logits must have shape [B,1,H,W].")

        if ignore_index is not None:
            valid = (target != ignore_index)
        else:
            valid = torch.ones_like(target, dtype=torch.bool)

        tgt = (target > 0).long()  # mappe IDs auf {0,1}
        tgt = tgt.unsqueeze(1).float()        # [B,1,H,W]
        prob = prob * valid.unsqueeze(1)      # maske anwenden
        tgt  = tgt  * valid.unsqueeze(1)

        intersect = (prob * tgt).sum(dim=(0,2,3))
        denom     = prob.sum(dim=(0,2,3)) + tgt.sum(dim=(0,2,3))
        dice      = (2.0 * intersect + epsilon) / (denom + epsilon)  # [1]
        return 1.0 - dice.mean()