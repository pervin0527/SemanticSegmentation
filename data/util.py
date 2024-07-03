import cv2
import numpy as np

def mask_encoding(mask, red_threshold=0.6, green_threshold=0.6):
    # Convert BGR mask to HSV
    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Define thresholds for red and green channels in BGR
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])

    # Create masks for red and green colors
    red_mask1 = cv2.inRange(hsv_mask, red_lower, red_upper)
    red_mask2 = cv2.inRange(hsv_mask, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv_mask, green_lower, green_upper)

    # Initialize full_mask with zeros
    full_mask = np.zeros_like(red_mask)

    # Set red pixels to 1 and green pixels to 2
    full_mask[red_mask > 0] = 1
    full_mask[green_mask > 0] = 2

    return full_mask

# def mask_encoding(mask):  
#     hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    
#     # lower boundary RED color range values; Hue (0 - 10)
#     lower1 = np.array([0, 100, 20])
#     upper1 = np.array([10, 255, 255])
    
#     # upper boundary RED color range values; Hue (160 - 180)
#     lower2 = np.array([160,100,20])
#     upper2 = np.array([179,255,255])

#     lower_mask = cv2.inRange(hsv_mask, lower1, upper1)
#     upper_mask = cv2.inRange(hsv_mask, lower2, upper2)
#     red_mask = lower_mask + upper_mask
#     red_mask[red_mask != 0] = 2

#     # boundary RED color range values; Hue (36 - 70)
#     green_mask = cv2.inRange(hsv_mask, (36, 25, 25), (70, 255,255))
#     green_mask[green_mask != 0] = 1
#     full_mask = cv2.bitwise_or(red_mask, green_mask)
#     full_mask = full_mask.astype(np.uint8)

#     return full_mask


def mask_decoding(pred_mask):
    decoded_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    decoded_mask[pred_mask == 0] = [0, 0, 0]
    decoded_mask[pred_mask == 1] = [0, 255, 0] ## Green
    decoded_mask[pred_mask == 2] = [255, 0, 0] ## Red
    
    return decoded_mask


def get_bbox_from_mask(encoded_mask):
    classes = [1, 2]
    bounding_boxes_dict = {}

    for cls in classes:
        binary_mask = (encoded_mask == cls).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xmin = x
            ymin = y
            xmax = x + w
            ymax = y + h
            bounding_boxes.append((xmin, ymin, xmax, ymax))
        
        bounding_boxes_dict[cls] = bounding_boxes
    
    return bounding_boxes_dict