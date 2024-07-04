from torch.nn import functional as F

def mean_dice_coefficient(predicted, targets, num_classes, smooth=1):
    # predicted: (N, H, W)
    # targets: (N, H, W)
    
    # Ensure that predicted and targets are within valid range
    assert predicted.max() < num_classes, f"Predicted max value {predicted.max()} exceeds num_classes {num_classes}"
    assert predicted.min() >= 0, f"Predicted min value {predicted.min()} is less than 0"
    assert targets.max() < num_classes, f"Targets max value {targets.max()} exceeds num_classes {num_classes}"
    assert targets.min() >= 0, f"Targets min value {targets.min()} is less than 0"

    predicted = F.one_hot(predicted, num_classes).permute(0, 3, 1, 2).float()  # One-hot encode the prediction and rearrange dimensions
    targets = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # One-hot encode the target and rearrange dimensions

    predicted = predicted.contiguous()
    targets = targets.contiguous()

    intersection = (predicted * targets).sum(dim=(2, 3))  # Intersection part of the Dice coefficient
    union = predicted.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))  # Union part of the Dice coefficient

    dice_coefficient = ((2. * intersection + smooth) / (union + smooth)).mean(dim=1)  # Dice coefficient per class
    mean_dice = dice_coefficient.mean()  # Mean Dice coefficient across all classes
    
    return mean_dice.item()  # Convert to a standard Python float for logging