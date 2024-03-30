import cv2
import numpy as np

from torch.utils.data import Dataset

from data.util import mask_encoding
from data.augmentation import basic_transform, apply_transform

class BKAIDataset(Dataset):
    CLASSES = ["background", "non-neoplastic polyps", "neoplastic polyps"]
    COLORMAP = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, args, feature_extractor=None, image_set="train"):
        self.args = args
        self.feature_extractor = feature_extractor
        
        self.data_dir = args.data_dir
        self.image_dir = f"{self.data_dir}/train/train"
        self.mask_dir = f"{self.data_dir}/train_gt/train_gt"
        self.transform = basic_transform(True if image_set == "train" or image_set == "trainval" else False, img_size=args.img_size)

        with open(f"{self.data_dir}/files/{image_set}.txt", 'r') as f:
            self.total_files = [line.strip() for line in f.readlines()]


    def __len__(self):
        return len(self.total_files)
    

    def get_img_mask(self, file_name):
        image_path = f"{self.image_dir}/{file_name}.jpeg"
        mask_path = f"{self.mask_dir}/{file_name}.jpeg"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)    

        return image, mask


    def __getitem__(self, idx):
        file_name = self.total_files[idx]
        image, mask = self.get_img_mask(file_name)
        batch_image, batch_mask = apply_transform(image, mask, self.transform)

        batch_mask = mask_encoding(batch_mask)
        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(batch_image, batch_mask, return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()

            return encoded_inputs   
        
        else:
            return batch_image, batch_mask