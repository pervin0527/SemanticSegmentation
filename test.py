import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import torch
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig

from data.bkai import BKAIDataset
from utils.utils import Args

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 225] = 255
    pixels[pixels <= 225] = 0
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def rle2mask(mask_rle, shape=(3,3)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)

    r = {'ids': ids, 'strings': strings,}
    return r

def main(args):
    # model_config = SegformerConfig.from_pretrained(args.pretrained_model_name,
    #                                                id2label=args.id2label, 
    #                                                label2id=args.label2id,
    #                                                num_labels=len(args.classes),
    #                                                image_size=args.img_size,
    #                                                num_encoder_blocks=args.num_encoder_blocks,
    #                                                drop_path_rate=args.drop_path_rate,
    #                                                hidden_dropout_prob=args.hidden_dropout_prob,
    #                                                classifier_dropout_prob=args.classifier_dropout_prob,
    #                                                attention_probs_dropout_prob=args.attention_probs_dropout_prob,
    #                                                semantic_loss_ignore_index=args.semantic_loss_ignore_index)

    # model_config.save_pretrained(f'{args.save_dir}')
    # feature_extractor = SegformerImageProcessor.from_pretrained(args.pretrained_model_name, do_reduce_labels=args.do_reduce_labels)
    # model = SegformerForSemanticSegmentation.from_pretrained(f"{args.save_dir}/weights/best.pt",
    #                                                          config=model_config,
    #                                                          ignore_mismatched_sizes=True)
    
    model_config = SegformerConfig(
        id2label=args.id2label, 
        label2id=args.label2id,
        num_labels=len(args.classes),
        image_size=args.img_size,
        num_encoder_blocks=args.num_encoder_blocks,
        drop_path_rate=args.drop_path_rate,
        hidden_dropout_prob=args.hidden_dropout_prob,
        classifier_dropout_prob=args.classifier_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        semantic_loss_ignore_index=args.semantic_loss_ignore_index
    )
    model_config.save_pretrained(f'{args.save_dir}')
    feature_extractor = SegformerImageProcessor(do_reduce_labels=args.do_reduce_labels)
    model = SegformerForSemanticSegmentation(config=model_config)
    model.to(args.device)
    
    files = sorted(glob(f"{args.data_dir}/test/*.jpeg"))
    model.eval()
    with torch.no_grad():
        for file in tqdm(files, desc="Test"):
            file_name = file.split('/')[-1]
            o_image = cv2.imread(file)
            o_image = cv2.cvtColor(o_image, cv2.COLOR_BGR2RGB)
            oh, ow, _ = o_image.shape

            image = cv2.resize(o_image, (args.img_size, args.img_size))
            inputs = feature_extractor(image, return_tensors="pt")
            outputs = model(**inputs.to(args.device))
            logits = outputs.logits.cpu()

            upsampled_logits = F.interpolate(logits, size=(oh, ow), mode="bilinear", align_corners=False) ## mode="nearest"
            seg = upsampled_logits.argmax(dim=1)[0]
            
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(args.colormap):
                color_seg[seg == label, :] = np.array(color)

            color_seg = cv2.cvtColor(color_seg, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{args.save_dir}/test/{file_name}", color_seg)

    MASK_DIR_PATH = f"{args.save_dir}/test"
    dir = MASK_DIR_PATH
    res = mask2string(dir)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']

    df.to_csv(f'{args.save_dir}/output.csv', index=False)


if __name__ == "__main__":
    saved_dir = "/home/pervinco/SemanticSegmentation/runs/2024_07_01_23_21_13"

    args = Args(f"{saved_dir}/config.yaml", is_train=False)
    args.save_dir = saved_dir

    args.num_workers = os.cpu_count()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.classes = BKAIDataset.CLASSES
    args.colormap = BKAIDataset.COLORMAP

    id2label = {idx: label for idx, label in enumerate(args.classes)}
    label2id = {label: idx for idx, label in id2label.items()}
    args.id2label = id2label
    args.label2id = label2id

    main(args)