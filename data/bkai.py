import cv2
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from data.augmentation import get_transform


def mask_encoding(mask):  
    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    
    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255])

    lower_mask = cv2.inRange(hsv_mask, lower1, upper1)
    upper_mask = cv2.inRange(hsv_mask, lower2, upper2)
    red_mask = lower_mask + upper_mask
    red_mask[red_mask != 0] = 2

    # boundary RED color range values; Hue (36 - 70)
    green_mask = cv2.inRange(hsv_mask, (36, 25, 25), (70, 255,255))
    green_mask[green_mask != 0] = 1
    full_mask = cv2.bitwise_or(red_mask, green_mask)
    full_mask = full_mask.astype(np.uint8)

    return full_mask


def mask_decoding(pred_mask):
    decoded_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    decoded_mask[pred_mask == 0] = [0, 0, 0]
    decoded_mask[pred_mask == 1] = [0, 255, 0] ## Green
    decoded_mask[pred_mask == 2] = [255, 0, 0] ## Red
    
    return decoded_mask


class BKAIDataset(Dataset):
    CLASSES = ["background", "non-neoplastic polyps", "neoplastic polyps"]
    COLORMAP = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, args, feature_extractor=None, image_set="train"):
        self.args = args
        self.feature_extractor = feature_extractor
        
        self.data_dir = args.data_dir
        self.image_dir = f"{self.data_dir}/train/train"
        self.mask_dir = f"{self.data_dir}/train_gt/train_gt"
        self.transform = get_transform(True if image_set == "train" or image_set == "trainval" else False, img_size=args.img_size)

        with open(f"{self.data_dir}/{image_set}.txt", 'r') as f:
            self.total_files = [line.strip() for line in f.readlines()]


    def __len__(self):
        return len(self.total_files)
    

    def __getitem__(self, idx):
        file_name = self.total_files[idx]
        image_path = f"{self.image_dir}/{file_name}.jpeg"
        mask_path = f"{self.mask_dir}/{file_name}.jpeg"

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        
        transformed = self.transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']
        mask = mask_encoding(mask)
        

        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(image, mask, return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()

            return encoded_inputs   
        
        else:
            return image, mask