from src.dinov2_custom import Segmenter

import glob
import os
import cv2
from matplotlib.patches import Rectangle as rect
from tqdm import tqdm
import numpy as np
import pickle
import time
import torch
from typing import Tuple as tuple
from typing import List as list
from typing import Dict as dict
torch.cuda.empty_cache()

def get_data_num(path_str: str) -> int:
    return int(path_str.split('/')[-1].split('.')[0])

def get_image(path: str) -> np.ndarray:
    image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return image

def get_images(dir: str) -> dict[int, np.ndarray]:
    pattern = os.path.join(dir, '*.png')
    file_list = glob.glob(pattern)
    
    images = {}
    for file_path in file_list:
        file_path = file_path.replace('\\', '/')
        image = get_image(file_path)
        
        images[get_data_num(file_path)] = image
    
    return images

def get_label(path: str) -> np.ndarray:
    """
    i = 0: top-left (x, y)
    i = 1: (width, height)
    """
    
    file = open(path, 'r')
    content = file.read().split()
    vals = np.array([eval(x) for x in content[1:]])
    coords = np.zeros((4, 2))
    for i in range(4):
        coords[i] = [vals[i*2], vals[i*2+1]]
        
    label = np.array([coords[0], coords[2] - coords[0]])
    
    return label

def get_labels(dir: str) -> dict[int, np.ndarray]:
    pattern = os.path.join(dir, '*.txt')
    file_list = glob.glob(pattern)
    
    labels = {}
    for file_path in file_list:
        file_path = file_path.replace('\\', '/')
        label = get_label(file_path)
        
        labels[get_data_num(file_path)] = label
    
    return labels

def get_bboxes(dir) -> dict[int, tuple[int, int, int]]:
    pattern = os.path.join(dir, '*.txt')
    file_list = glob.glob(pattern)
    
    bboxes = {}
    for file_path in file_list:
        file_path = file_path.replace('\\', '/')
        
        file = open(file_path, 'r')
        content = file.read().split()
        vals = np.array([eval(x) for x in content])

        bboxes[get_data_num(file_path)] = ((vals[0], vals[1]), vals[2], vals[3])
    
    return bboxes

def get_bbox(image:np.ndarray, label: np.ndarray) -> tuple[tuple[int, int], int, int]:
    img_height, img_width, _ = image.shape
    
    xy = (round(img_width * label[0][0]), round(img_height * label[0][1]))
    bbox_width = round(img_width * label[1][0])
    bbox_height = round(img_height * label[1][1])
    
    bbox = (xy, bbox_width, bbox_height)
    return bbox
    
def prepare_data(paths: dict[str, str], type: str) -> (dict, dict):
    images = get_images(paths[type]['images'])
    labels = get_labels(paths[type]['labels'])
    
    bboxes = {}
    for key in images.keys():
        bboxes[key] = get_bbox(images[key], labels[key])
    
    return images, bboxes, len(images)

def generate_masks(segmenter: Segmenter, images: dict[int, np.ndarray]) -> (dict, float):
    masks = {}
    
    start = time.time()
    for key in tqdm(images.keys()):
        image = images[key]
        
        generated_masks = segmenter.generate_masks(image)
        masks[key] = generated_masks
    end = time.time()
    
    return masks, end - start

def generate_overlays(segmenter: Segmenter, masks: list[dict]) -> dict:
    overlays = {}
    for key in masks.keys():
        overlay = segmenter.prepare_masks(masks[key])
        overlays[key] = overlay
        
    return overlays

# def generate_masks_and_overlays(segmenter: Segmenter, images: dict[int, np.ndarray]) -> (dict, dict):
#     masks = {}
#     overlays = {}
    
#     for key in tqdm(images.keys()):
#         image = images[key]
        
#         generated_masks = segmenter.generate_masks(image)
#         overlay = segmenter.prepare_masks(generated_masks)
        
#         masks[key] = generated_masks
#         overlays[key] = overlay
    
#     return masks, overlays

def save_masks_and_overlays(masks: dict, t: float, overlays: dict, paths: dict[str, str], type: str, seg_model_size: str) -> None:
    with open(paths[type]['masks'] + 'masks_' + seg_model_size + '.pkl', 'wb') as file:
        pickle.dump(masks, file)
    file.close()
        
    with open(paths[type]['masks'] + 'overlays_' + seg_model_size + '.pkl', 'wb') as file:
        pickle.dump(overlays, file)
    file.close()
    
    with open(paths[type]['masks'] + 'time_' + seg_model_size + '.txt', 'w') as file:
        file.write(str(t))
    file.close()
    
    torch.cuda.empty_cache()
        
def prepare_masks_and_overlays(images: dict[int, np.ndarray], paths: dict[str, str], type: str, seg_model_size: str) -> (dict, dict):
    segmenter = Segmenter(model_size=seg_model_size)
    masks, t = generate_masks(segmenter, images)
    overlays = generate_overlays(segmenter, masks)
    save_masks_and_overlays(masks, t, overlays, paths, type, seg_model_size)
    return masks, overlays

def load_masks_and_overlays(paths: dict[str, str], type: str, seg_model_size: str) -> (dict, dict):
    try:
        with open(paths[type]['masks'] + 'masks_' + seg_model_size + '.pkl', 'rb') as file:
            masks = pickle.load(file)
        file.close()
        
        with open(paths[type]['masks'] + 'overlays_' + seg_model_size + '.pkl', 'rb') as file:
            overlays = pickle.load(file)
        file.close()
        
        return masks, overlays
    except FileNotFoundError:
        return None, None

def make_bbox(bbox: tuple[tuple[int, int], int, int], linewidth=1, edgecolor='r', facecolor='none') -> rect:
    xy, width, height = bbox
    return rect(xy, width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)

def plot_image(image: np.ndarray, bbox: rect, ax, within: bool = False, title: str = None) -> None:
    ax.imshow(image)
    if bbox is not None:
        edgecolor = 'g' if within else 'r'
        ax.add_patch(make_bbox(bbox, edgecolor=edgecolor))
    ax.set_axis_off()
    ax.set_title(title)

def mask_within_bbox(mask: dict, bbox: tuple[tuple[int, int], int, int], thresh: float = 0.5) -> bool:
    mask_img = np.array(mask['segmentation'], dtype=np.uint8)
    tot_area = np.sum(mask_img)
    
    (x, y), w, h = bbox
    
    area_in_bbox = 0
    for dw in range(w):
        for dh in range(h):
            if mask_img[y + dh, x + dw] == 1:
                area_in_bbox += 1

    return area_in_bbox / tot_area > thresh, area_in_bbox / tot_area

def find_mask_in_bbox(masks: list[dict], bbox: tuple[tuple[int, int], int, int], thresh: float = 0.5) -> list[dict]:
    found_masks = []
    for mask in masks:
        within, ratio = mask_within_bbox(mask, bbox, thresh)
        if within:
            found_masks.append((mask, ratio))
    
    return found_masks

def get_correct_masked_images(images: dict[int, np.ndarray], all_masks: dict[str, list[dict]], bboxes: dict[int, tuple[tuple[int, int], int, int]], thresh: float = 0.5) -> dict[int, np.ndarray]:
    correct_masked_images = {}
    for key in tqdm(images.keys()):
        for seg_model_size, masks in all_masks.items():
            if len(masks):
                possible_masks = find_mask_in_bbox(masks[key], bboxes[key], thresh)
                if len(possible_masks) > 0:
                    correct_masked_images[key] = seg_model_size
        
    return correct_masked_images

def get_all_masks(paths: dict[str, str], type: str) -> dict[str, list[dict]]:
    all_masks = {'mobile': {}, 
                 'small':  {},
                 'medium': {},
                 'large':  {}}

    for seg_model_size in all_masks.keys():
        masks, _ = load_masks_and_overlays(paths, type, seg_model_size)
        if masks is not None:
            all_masks[seg_model_size] = masks
    
    return all_masks

def save_correct_masked_ids(correct_masked_images: dict[int, str], type: str) -> None:
    with open(f'data/ir_data/{type}/' + 'correct_masked_ids.txt', 'w') as file:
        file.write('id, seg_model_size\n')
        for key, seg_model_size in correct_masked_images.items():
            file.write(f'{key}, {seg_model_size}\n')
    file.close()

def load_correct_masked_ids(type: str) -> dict[int, str]:
    correct_masked_images = {}
    with open(f'data/ir_data/{type}/' + 'correct_masked_ids.txt', 'r') as file:
        file.readline()
        for line in file:
            key, seg_model_size = line.split(',')
            correct_masked_images[int(key)] = seg_model_size.strip()
    file.close()
    
    return correct_masked_images