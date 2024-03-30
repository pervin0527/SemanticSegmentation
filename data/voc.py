import cv2
import numpy as np

from torch.utils.data import Dataset
from data.augmentation import basic_transform


class VOCDataset(Dataset):
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "potted plant", "sheep", "sofa", "train",
               "tv/monitor"]
    
    COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
                [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

    def __init__(self, args, feature_extractor=None, image_set="train", year=2012):
        self.args = args
        self.data_dir = f"{args.data_dir}/VOC{year}"
        self.image_dir = f"{self.data_dir}/JPEGImages"
        self.mask_dir = f"{self.data_dir}/SegmentationClass"

        self.feature_extractor = feature_extractor
        self.transform = basic_transform(True if image_set == "train" or image_set == "trainval" else False, img_size=args.img_size)

        with open(f"{self.data_dir}/ImageSets/Segmentation/{image_set}.txt", "r") as file:
            self.file_names = file.read().splitlines()


    def __len__(self):
        return len(self.file_names)
    

    def convert_to_segmentation_mask(self, mask, binary=True):
        height, width = mask.shape[:2]

        if binary:
            segmentation_mask = np.zeros((height, width), dtype=np.uint8)

            for label_index, label_color in enumerate(self.args.COLORMAP):
                match = np.all(mask == np.array(label_color), axis=-1)
                segmentation_mask[match] = label_index
        else:
            segmentation_mask = np.zeros((height, width, len(self.args.COLORMAP)), dtype=np.float32) ## [height, width, num_classes]
            for label_index, label in enumerate(self.args.COLORMAP):
                segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)

        return segmentation_mask


    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_file, mask_file = f"{self.image_dir}/{file_name}.jpg", f"{self.mask_dir}/{file_name}.png"

        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.convert_to_segmentation_mask(mask, binary=True)

        transformed = self.transform(image=image, mask=mask)

        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(transformed['image'], transformed['mask'], return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()

            return encoded_inputs   
        
        else:
            return transformed["image"], transformed["mask"]