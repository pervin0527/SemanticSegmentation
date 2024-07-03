import cv2
import copy
import random
import numpy as np
import albumentations as A

from data.util import mask_encoding, get_bbox_from_mask


def basic_transform(is_train, img_size):
    if is_train:
        transform = A.Compose([
            A.OneOf([
                A.Resize(img_size, img_size, p=0.25),
                A.RandomSizedBBoxSafeCrop(img_size, img_size, p=0.25),
                A.ShiftScaleRotate(p=0.25, border_mode=0, shift_limit=0.15, scale_limit=0.15, rotate_limit=90),
                A.Compose([
                    A.RandomSizedBBoxSafeCrop(height=img_size//2, width=img_size//2, p=1),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, p=1)], 
                p=0.25),
            ], p=1),

            A.OneOf([
                A.HorizontalFlip(p=0.33),
                A.VerticalFlip(p=0.33),
                A.ElasticTransform(p=0.33, border_mode=0),
            ], p=1),
            
            A.OneOf([
                A.Blur(), 
                A.GaussianBlur(), 
                A.GlassBlur(), 
                A.MotionBlur(), 
                A.GaussNoise(), 
                A.Sharpen(), 
                A.MedianBlur(), 
                A.MultiplicativeNoise()
            ], p=0.5),

            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma (gamma_limit=(70, 130), eps=None, always_apply=False, p=0.2),
            A.RGBShift(p=0.3, r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.15, brightness_coeff=1.5, p=0.09),
            A.RandomShadow(p=0.1),
            A.Resize(img_size, img_size, p=1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    else:
        transform = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1, always_apply=True),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return transform


def apply_transform(i, m, b, l, transform):
    image, mask, bboxes, labels = copy.deepcopy(i), copy.deepcopy(m), copy.deepcopy(b), copy.deepcopy(l)
    transformed = transform(image=image, mask=mask, bboxes=bboxes, labels=labels)
    t_image, t_mask, t_bboxes, t_labels = transformed['image'], transformed['mask'], transformed['bboxes'], transformed['labels'] 

    return t_image, t_mask, t_bboxes, t_labels


def mosaic(piecies, size):
    h, w = size, size
    mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
    mosaic_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # cx, cy = random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4)
    cx, cy = w // 2, h // 2
    
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    for i, index in enumerate(indices):
        piece_image, piece_mask = piecies[index][0], piecies[index][1]
        
        if i == 0:
            mosaic_img[:cy, :cx] = cv2.resize(piece_image, (cx, cy))
            mosaic_mask[:cy, :cx] = cv2.resize(piece_mask, (cx, cy))
        elif i == 1:
            mosaic_img[:cy, cx:] = cv2.resize(piece_image, (w-cx, cy))
            mosaic_mask[:cy, cx:] = cv2.resize(piece_mask, (w-cx, cy))
        elif i == 2:
            mosaic_img[cy:, :cx] = cv2.resize(piece_image, (cx, h-cy))
            mosaic_mask[cy:, :cx] = cv2.resize(piece_mask, (cx, h-cy))
        elif i == 3:
            mosaic_img[cy:, cx:] = cv2.resize(piece_image, (w-cx, h-cy))
            mosaic_mask[cy:, cx:] = cv2.resize(piece_mask, (w-cx, h-cy))
    
    return mosaic_img, mosaic_mask

## spatially_exclusive_pasting
def sep(image, mask, img_size, alpha=0.7, iterations=10):
    augmentation = A.Compose([A.RandomBrightnessContrast(p=0.3),
                              A.GaussianBlur(p=0.2),])    

    target_image, target_mask = copy.deepcopy(image), copy.deepcopy(mask)
    L_gray = cv2.cvtColor(target_mask, cv2.COLOR_BGR2GRAY)
    
    encoded_mask = mask_encoding(mask)
    bounding_boxes = get_bbox_from_mask(encoded_mask)
    
    for cls, bboxes in bounding_boxes.items():
        for idx, bbox in enumerate(bboxes):
            xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            Lf_gray = L_gray[ymin:ymax, xmin:xmax]
            If = target_image[ymin:ymax, xmin:xmax]
            Lf_color = target_mask[ymin:ymax, xmin:xmax]

            # Augmentation
            augmented = augmentation(image=If, mask=Lf_color)
            If_augmented = augmented['image']
            Lf_color_augmented = augmented['mask']

            M = np.random.rand(*target_image.shape[:2])
            M[L_gray == 1] = float('inf')
            
            height, width = ymax - ymin, xmax - xmin
            
            for _ in range(iterations):
                px, py = np.unravel_index(M.argmin(), M.shape)        
                candidate_area = (slice(px, px + height), slice(py, py + width))
                
                if candidate_area[0].stop > target_image.shape[0] or candidate_area[1].stop > target_image.shape[1]:
                    M[px, py] = float('inf')
                    continue
                
                if np.any(L_gray[candidate_area] & Lf_gray):
                    M[candidate_area] = float('inf')
                    continue
                
                target_image[candidate_area] = alpha * target_image[candidate_area] + (1 - alpha) * If_augmented
                target_mask[candidate_area] = alpha * target_mask[candidate_area] + (1 - alpha) * Lf_color_augmented
                L_gray[candidate_area] = cv2.cvtColor(target_mask[candidate_area], cv2.COLOR_BGR2GRAY)
                
                M[candidate_area] = float('inf')
                
                kernel = np.ones((3, 3), np.float32) / 9
                M = cv2.filter2D(M, -1, kernel)

    target_image, target_mask = cv2.resize(target_image, (img_size, img_size)), cv2.resize(target_mask, (img_size, img_size))
    return target_image, target_mask


def mixup(foreground_image, background_image, img_size, alpha):
    image1, image2 = copy.deepcopy(foreground_image), copy.deepcopy(background_image)

    height, width = image1.shape[:2]
    background_transform = A.Compose([A.RandomRotate90(p=0.5),
                                      A.VerticalFlip(p=0.3),
                                      A.HorizontalFlip(p=0.3),
                                      A.RandomBrightnessContrast(p=0.6),
                                      A.RGBShift(p=0.3),
                                      A.Resize(img_size, img_size, p=1, always_apply=True)
                                      ])
    
    transformed = background_transform(image=image2)
    transformed_image = transformed["image"]

    mixed_image = cv2.addWeighted(image1, alpha, transformed_image, 1 - alpha, 0)   
    
    return mixed_image


def get_bg_image(bg_files):
    bg_idx = random.randint(0, len(bg_files) - 1)
    background_image = cv2.imread(bg_files[bg_idx])
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    return background_image