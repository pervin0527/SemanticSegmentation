import os
import cv2
import numpy as np

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from data.augmentation import get_transform

class COCODataset(Dataset):
    def __init__(self, args, feature_extractor=None, image_set='train'):
        self.args = args
        self.root = f"{args.data_dir}"
        self.image_dir = f"{args.data_dir}/{image_set}2017"
        self.coco = COCO(f"{args.data_dir}/annotations/instances_{image_set}2017.json")
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.feature_extractor = feature_extractor
        self.transform = get_transform(True if image_set == "train" else False, args.img_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.image_dir, path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        # 모든 클래스 인덱스로 채워진 마스크를 저장할 배열을 생성합니다. 초기값은 0 (배경)입니다.
        mask = np.zeros((height, width), dtype=np.uint8)
        
        bboxes = []
        for obj in coco_annotation:
            # pycocotools에서 제공하는 annToMask 메서드를 사용하여 마스크를 생성합니다.
            obj_mask = self.coco.annToMask(obj)
            # 마스크에 클래스 인덱스를 적용합니다. 마스크의 해당 영역을 클래스 인덱스로 채웁니다.
            mask[obj_mask == 1] = obj['category_id']
            # 바운딩 박스 정보를 저장합니다. 여기에는 클래스 인덱스와 객체의 인덱스도 포함됩니다.
            bboxes.append(obj['bbox'] + [obj['category_id']] + [obj['id']])

        transformed = self.transform(image=image, mask=mask)

        if self.feature_extractor is not None:
            encoded_inputs = self.feature_extractor(transformed['image'], transformed['mask'], return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()

            return encoded_inputs   
        
        else:
            return transformed["image"], transformed["mask"]