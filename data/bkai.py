import os
import cv2
import random
import numpy as np

from glob import glob
from torch.utils.data import Dataset

from data.util import mask_encoding
from data.augmentation import basic_transform, apply_transform, sep, mosaic, mixup, get_bg_image

class BKAIDataset(Dataset):
    CLASSES = ["background", "non-neoplastic polyps", "neoplastic polyps"]
    COLORMAP = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]

    def __init__(self, args, feature_extractor=None, image_set="train"):
        self.args = args
        self.feature_extractor = feature_extractor
        self.is_train = True if image_set == "train" else False
        
        self.data_dir = args.data_dir
        self.image_dir = f"{self.data_dir}/train/train"
        self.mask_dir = f"{self.data_dir}/train_gt/train_gt"
        self.bbox_dir = f"{self.data_dir}/train_boxes/train_boxes"
        self.transform = basic_transform(is_train=self.is_train, img_size=args.img_size)

        with open(f"{self.data_dir}/files/{image_set}.txt", 'r') as f:
            self.total_files = [line.strip() for line in f.readlines()]

        self.bg_files = glob(f"{self.data_dir}/background/0_normal/*.jpg")

    def __len__(self):
        return len(self.total_files)

    def get_img_mask(self, file_name):
        image_path = f"{self.image_dir}/{file_name}.jpeg"
        mask_path = f"{self.mask_dir}/{file_name}.jpeg"
        bbox_path = f"{self.bbox_dir}/{file_name}.txt"
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        
        labels = []
        bboxes = []
        if os.path.exists(bbox_path):
            with open(bbox_path, 'r') as f:
                for line in f:
                    label, xmin, ymin, xmax, ymax = map(int, line.strip().split())
                    labels.append(label)
                    bboxes.append((xmin, ymin, xmax, ymax))
                    
        return image, mask, bboxes, labels


    def __getitem__(self, idx):
        if self.is_train:
            p = random.random()
            if p < 0.3:
                file_name = self.total_files[idx]
                image, mask, bboxes, labels = self.get_img_mask(file_name)
                batch_image, batch_mask, batch_bboxes, batch_labels = apply_transform(image, mask, bboxes, labels, self.transform)

                if random.random() > 0.7:
                    background_image = get_bg_image(self.bg_files)
                    batch_image = mixup(batch_image, background_image, img_size=self.args.img_size, alpha=random.uniform(self.args.mixup_alpha, self.args.mixup_alpha + 0.3))

            elif 0.3 < p <= 0.6:
                piecies = []
                while len(piecies) < 4:
                    i = random.randint(0, len(self.total_files)-1)
                    file_name = self.total_files[i]
                    image, mask, bboxes, labels = self.get_img_mask(file_name)

                    if random.random() > 0.5:
                        piece_image, piece_mask, batch_bboxes, batch_labels = apply_transform(image, mask, bboxes, labels, self.transform)
                    else:
                        piece_image, piece_mask = sep(image, mask, self.args.img_size, alpha=random.uniform(self.args.spatial_alpha, self.args.spatial_alpha + 0.2))

                    piecies.append([piece_image, piece_mask])

                batch_image, batch_mask = mosaic(piecies, size=self.args.img_size)
                if random.random() > 0.7:
                    background_image = get_bg_image(self.bg_files)
                    piece_image = mixup(batch_image, background_image, img_size=self.args.img_size, alpha=random.uniform(self.args.mixup_alpha, self.args.mixup_alpha + 0.3))

            elif 0.6 < p <= 1:
                file_name = self.total_files[idx]
                image, mask, bboxes, labels = self.get_img_mask(file_name)
                batch_image, batch_mask = sep(image, mask, self.args.img_size, alpha=random.uniform(self.args.spatial_alpha, self.args.spatial_alpha + 0.2))

                if random.random() > 0.7:
                    background_image = get_bg_image(self.bg_files)
                    batch_image = mixup(batch_image, background_image, img_size=self.args.img_size, alpha=random.uniform(self.args.mixup_alpha, self.args.mixup_alpha + 0.3))
                
        else:
            file_name = self.total_files[idx]
            image, mask, bboxes, labels = self.get_img_mask(file_name)
            batch_image, batch_mask, batch_bboxes, batch_labels = apply_transform(image, mask, bboxes, labels, self.transform)

        batch_mask = mask_encoding(batch_mask)
        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(batch_image, batch_mask, return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()

            return encoded_inputs   
        
        else:
            return batch_image, batch_mask