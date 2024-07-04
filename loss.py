import torch
import evaluate
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1):
        # inputs: (N, num_classes, H, W)
        # targets: (N, H, W)
        
        inputs = F.softmax(inputs, dim=1)  # Apply softmax to the inputs to get probabilities
        targets = F.one_hot(targets, self.num_classes)  # One-hot encode the target
        
        if inputs.dim() == 4:
            targets = targets.permute(0, 3, 1, 2)  # (N, H, W, num_classes) -> (N, num_classes, H, W)
        
        inputs = inputs.contiguous()
        targets = targets.contiguous()

        intersection = (inputs * targets).sum(dim=(2, 3))  # Intersection part of the Dice coefficient
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Union part of the Dice coefficient

        dice_loss = 1 - ((2. * intersection + smooth) / (union + smooth)).mean(dim=1)  # Dice loss per class
        return dice_loss.mean()  # Mean Dice loss across all classes
    

class FocalLoss(nn.Module):
    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super().__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()
            prob = prob.view(-1, prob.size(-1))

        ori_shp = target.shape
        target = target.view(-1, 1)
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        prob = prob.gather(1, target).view(-1) + self.smooth
        logpt = torch.log(prob)
        
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss