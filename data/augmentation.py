import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# transform = A.Compose([
#     A.Resize(height=img_size, width=img_size),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
#     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.8),
#     A.RandomBrightnessContrast(p=0.5),
#     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     # ToTensorV2(),
# ])

def get_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.Resize(img_size, img_size, p=1, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
            A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            
            A.OneOf([A.Blur(), A.GaussianBlur(), A.GlassBlur(), A.MotionBlur(), A.GaussNoise(), A.Sharpen(), A.MedianBlur(), A.MultiplicativeNoise()]),

            A.CoarseDropout(p=0.2, max_height=35, max_width=35, fill_value=0, mask_fill_value=0),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
            A.RandomShadow(p=0.1),
            A.ShiftScaleRotate(p=0.45, border_mode=cv2.BORDER_CONSTANT, shift_limit=0.15, scale_limit=0.15, rotate_limit=180),
            A.RandomCrop(img_size, img_size)])

    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
        ])

    return transform