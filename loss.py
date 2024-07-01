import torch
import evaluate
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        dice = 0

        logits = torch.softmax(logits, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        
        for i in range(num_classes):
            iflat = logits[:, i].contiguous().view(-1)
            tflat = targets[:, i].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            
            dice += (2. * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)

        return 1 - dice / num_classes



class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        jaccard = 0

        logits = torch.softmax(logits, dim=1)
        targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).contiguous()
        for i in range(num_classes):
            iflat = logits[:, i].contiguous().view(-1)
            tflat = targets[:, i].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            union = iflat.sum() + tflat.sum() - intersection
            
            jaccard += (intersection + self.smooth) / (union + self.smooth)

        return 1 - jaccard / num_classes


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
    

def compute_mean_iou(model, dataloader, device, num_labels, ignore_index=255):
    model.eval()
    metric = evaluate.load("mean_dice")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Mean IoU", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            upsampled_logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            # Update metric for each batch
            metric.add_batch(predictions=predicted, references=labels)

    # Compute the mean IoU over all batches
    metrics = metric.compute(num_labels=num_labels, ignore_index=ignore_index)
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    detailed_metrics = {
        "mean_iou": metrics["mean_iou"],
        **{f"accuracy_class_{i}": acc for i, acc in enumerate(per_category_accuracy)},
        **{f"iou_class_{i}": iou for i, iou in enumerate(per_category_iou)}
    }

    return detailed_metrics


def dice_coefficient(preds, labels, num_classes):
    """
    Calculate the Dice coefficient for multiple classes.
    preds: Predicted tensors (N, H, W) with values indicating class predictions.
    labels: Ground truth tensors (N, H, W) with actual class labels.
    num_classes: The number of classes.
    """
    dice_scores = []
    epsilon = 1e-7

    for class_id in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (preds == class_id)
        label_mask = (labels == class_id)
        
        intersection = (pred_mask & label_mask).sum()
        union = pred_mask.sum() + label_mask.sum()
        
        dice_score = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice_score)

    # Calculate the average Dice coefficient across all classes
    mean_dice = sum(dice_scores) / num_classes
    return mean_dice

def compute_mean_dice_coefficient_score(model, dataloader, device, num_classes):
    model.eval()
    total_dice = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Mean Dice Coefficient", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            upsampled_logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            predicted = upsampled_logits.argmax(dim=1)

            dice_score = dice_coefficient(predicted.cpu(), labels.cpu(), num_classes)
            total_dice += dice_score.item()
            count += 1

    mean_dice = total_dice / count
    return mean_dice