from patchify import patchify
from transformers import SamModel, SamProcessor, SamConfig, ViTFeatureExtractor, ViTForImageClassification
import torch
from datasets import Dataset
from PIL import Image
from matplotlib.patches import Rectangle
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import cv2 as cv
from typing import Tuple

THRESHOLD = 0.65
IoU_THRESHOLD = 0.5

def resize_image(img_arr: np.ndarray, w: int, h: int) -> np.ndarray:
    img = Image.fromarray(img_arr)
    img = img.resize((w, h))
    return np.array(img)


def prepare_image(image: np.ndarray, dims: tuple[int, int], patch_size: int, verbose=True) -> torch.Tensor:
    start_time = time.time()
    
    image = resize_image(image, dims[0], dims[1])
    patches = patchify(np.array(image), (patch_size, patch_size, 3), step=256)
    patches = np.squeeze(patches)
    
    end_time = time.time()
    
    if verbose:
        print(f"Time taken to prepare image: {end_time - start_time:.2f} seconds")
    
    return patches


def calculate_distance_with_all_neighbors(contours_info):
    coords = []
    for contour_info in contours_info:
        bbox = contour_info['bounding_box']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        x_mid = x + w // 2
        y_mid = y + h // 2
        coords.append((x_mid, y_mid))

    tot_dist = 0
    num_dists = 0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            tot_dist += dist
            num_dists += 1
    
    if num_dists == 0: return 1
    return tot_dist**2


def mask_within_bbox(mask, bbox: tuple[tuple[int, int], int, int], thresh: float = 0.5) -> bool:
    mask_img = np.array(mask, dtype=np.uint8)
    tot_area = np.sum(mask_img)
    
    x, y, w, h = bbox
    
    area_in_bbox = 0
    for dw in range(w):
        for dh in range(h):
            if mask_img[y + dh, x + dw] == 1:
                area_in_bbox += 1

    return area_in_bbox / tot_area >= thresh, area_in_bbox / tot_area


def joint_intersection(mask1, mask2, thresh) -> bool:
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if union.sum() == 0:
        return True, 0
    return np.sum(intersection) / np.sum(union) >= thresh, np.sum(intersection) / np.sum(union)
    

def plot_prediction(image: np.ndarray, prediction: np.ndarray, probability: np.ndarray, idx: int, bbox: tuple = None) -> None:
    if bbox:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fix, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot the first image on the left
    axes[0].imshow(np.array(image), cmap='gray')  # Assuming the first image is grayscale
    axes[0].set_title(idx)

    # Plot the second image on the right
    axes[1].imshow(probability)  # Assuming the second image is grayscale
    axes[1].set_title("Probability Map")

    # Plot the second image on the right
    axes[2].imshow(prediction, cmap='gray')  # Assuming the second image is grayscale
    axes[2].set_title("Prediction")
    
    if bbox:
        axes[3].imshow(prediction, cmap='gray') 
        axes[3].set_title("Prediction with bbox")
        
        x, y, w, h = bbox
        r = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[3].add_patch(r)

    # Hide axis ticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Display the images side by side
    plt.show()


class SegmentSat():
    def __init__(self, 
                 sam_model_path: str, 
                 vit_classifier_path: str, 
                 image_transform_dims: tuple[int, int] = (512, 512),
                 patch_size: int = 256,
                 sam_processor_path: str = None, 
                 vit_feature_extractor_path: str = None):
        
        self.image_transform_dims = image_transform_dims
        self.patch_size = patch_size
        
        self.device = self._get_device()

        self._load_sam_model_and_processor(sam_model_path, sam_processor_path)
        self._load_vit_classifier_and_feature_extractor(vit_classifier_path, vit_feature_extractor_path)
        
    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device
        
    def _load_sam_model_and_processor(self, 
                                      sam_model_path: str, 
                                      sam_processor_path: str = None):
        if sam_processor_path is None:
            sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        else:
            sam_processor = SamProcessor.from_pretrained(sam_processor_path)
        
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        sam_model = SamModel(config=model_config)
        checkpoint = torch.load(sam_model_path)
        sam_model.load_state_dict(checkpoint['model_state_dict'])
        sam_model.to(self.device)
        
        self.sam_processor = sam_processor
        self.sam_model = sam_model
    
    def _load_vit_classifier_and_feature_extractor(self, 
                                                   vit_classifier_path: str, 
                                                   vit_feature_extractor_path: str = None):
        if vit_feature_extractor_path is None:
            vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            vit_feature_extractor = ViTFeatureExtractor.from_pretrained(vit_feature_extractor_path)
        
        vit_classifier = ViTForImageClassification.from_pretrained(vit_classifier_path).to(self.device)
        vit_classifier.to(self.device)
        
        self.vit_feature_extractor = vit_feature_extractor
        self.vit_classifier = vit_classifier
    
    def _has_satellite(self, image):
        image_tensor = self.vit_feature_extractor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        logits = self.vit_classifier(image_tensor).logits
        prediction = logits.argmax(axis=-1)
        return bool(prediction == 1)
        
    def predict(self, image: torch.Tensor, thresh: float = THRESHOLD, verbose=False) -> Tuple[torch.Tensor, torch.Tensor]:
        
        original_dims = image.shape[:-1]
        
        start_time = time.time()
        self.sam_model.eval()
        
        patches = prepare_image(image, self.image_transform_dims, patch_size=self.patch_size, verbose=verbose)
        prob_patches = []
        orig_patches = []
        has_sats = []
        
        fig, axs1 = plt.subplots(patches.shape[0], patches.shape[1], figsize=(10, 10))
        fig, axs2 = plt.subplots(patches.shape[0], patches.shape[1], figsize=(10, 10))
        
        for i, row_patches in enumerate(patches):
            prob_patches.append([])
            has_sats.append([])
            orig_patches.append([])
            for j, patch in enumerate(row_patches):
                has_sat = self._has_satellite(patch)
                if verbose: print("has_sat", has_sat)
                has_sats[-1].append(has_sat)

                inputs = self.sam_processor(patch, return_tensors="pt")
                inputs = {k:v.to(self.device) for k,v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.sam_model(**inputs, multimask_output=False)
                single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
                single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
                
                orig_patches[-1].append(single_patch_prob)
                
                axs1[i, j].imshow(single_patch_prob)
                axs1[i, j].axis('off')  # Hide axes
                axs1[i, j].set_title(f"Has Satellite: {has_sat}")
                
                single_patch_prob = single_patch_prob / np.sum(single_patch_prob)
                
                # Gaussian blur
                single_patch_prob = cv.GaussianBlur(single_patch_prob, (7, 7), 0)
                
                axs2[i, j].imshow(single_patch_prob)
                axs2[i, j].axis('off')  # Hide axes
                
                prob_patches[-1].append(single_patch_prob)
        
        prob_patches = np.array(prob_patches)
        orig_patches = np.array(orig_patches)
        
        patch_contour_dists = []
        patch_contour_counts = []
        patch_contour_bboxes = []
        patch_contour_areas = []
        
        if verbose: 
            fig, axs = plt.subplots(patches.shape[0], patches.shape[1], figsize=(10, 10))
        
        for i in range(prob_patches.shape[0]):
            patch_contour_dists.append([])
            patch_contour_counts.append([])
            patch_contour_bboxes.append([])
            patch_contour_areas.append([])
            
            for j in range(prob_patches.shape[1]):
                patch = prob_patches[i][j]
                tot_dist_factor = patch.max()
                patch = patch / patch.max() * 255
                
                patch = patch.astype(np.uint8)
                patch = np.stack((patch, patch, patch), axis=2)
                imgray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                ret, t = cv.threshold(imgray, 127, 255, 0)
                contours, hierarchy = cv.findContours(t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                contours_info = []
                for contour in contours:
                    area = cv.contourArea(contour)
                    perimeter = cv.arcLength(contour, True)
                    x, y, w, h = cv.boundingRect(contour)
                    contours_info.append({
                        'area': area,
                        'perimeter': perimeter,
                        'bounding_box': {'x': x, 'y': y, 'width': w, 'height': h}
                    })
                    
                tot_dist = calculate_distance_with_all_neighbors(contours_info)

                tot_dist = tot_dist / (tot_dist_factor**2)
                print("tot_dist", tot_dist)
                
                patch_contour_dists[-1].append(tot_dist)
                patch_contour_counts[-1].append(len(contours))
                patch_contour_bboxes[-1].append([i['bounding_box'] for i in contours_info])
                patch_contour_areas[-1].append(sum([i['area'] for i in contours_info]))
                
                if verbose: 
                    cv.drawContours(patch, contours, -1, (0, 255, 0), 3)
                    axs[i, j].imshow(cv.cvtColor(patch, cv.COLOR_BGR2RGB))
                    axs[i, j].axis('off')  # Hide axes
        
        fig, axs3 = plt.subplots(patches.shape[0], patches.shape[1], figsize=(10, 10))
        
        patch_contour_counts = np.array(patch_contour_counts)
        patch_contour_areas = np.array(patch_contour_areas)
        patch_contour_dists = np.array(patch_contour_dists)
        dists_sum = sum([1/d for d in patch_contour_dists.flatten()])

        prob = None
        orig_prob = None
        for i in range(prob_patches.shape[0]):
            row_prob = None
            orig_row_prob = None
            for j in range(prob_patches.shape[1]):
                single_patch_prob = prob_patches[i][j]
                
                contour_mask = np.zeros(single_patch_prob.shape)
                for bbox in patch_contour_bboxes[i][j]:
                    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                    
                    contour_mask[y:y+h, x:x+w] = True
                
                single_patch_prob = np.where(contour_mask, single_patch_prob, 0)
                
                contour_factor = 1
                contour_factor = 1/(patch_contour_dists[i][j] * dists_sum)
                if patch_contour_counts[i][j] == 0 or patch_contour_areas[i][j] <= 1:
                    pass
                else:
                    contour_factor /= patch_contour_counts[i][j]
                    
                if not has_sats[i][j]:
                    contour_factor = 0
                
                if verbose: print("contour_factor", contour_factor)
                single_patch_prob = single_patch_prob * (contour_factor)
                
                axs3[i, j].imshow(single_patch_prob)
                axs3[i, j].axis('off')  # Hide axes
                axs3[i, j].set_title(f"c_patch: {round(contour_factor, 2)}")
                
                if row_prob is None:
                    row_prob = single_patch_prob
                else:
                    row_prob = np.concatenate((row_prob, single_patch_prob), axis=1)
                
                if orig_row_prob is None:
                    orig_row_prob = orig_patches[i][j]
                else:
                    orig_row_prob = np.concatenate((orig_row_prob, orig_patches[i][j]), axis=1)
                    
            if prob is None:
                prob = row_prob
            else:
                prob = np.concatenate((prob, row_prob), axis=0)
            
            if orig_prob is None:
                orig_prob = orig_row_prob
            else:
                orig_prob = np.concatenate((orig_prob, orig_row_prob), axis=0)
        
        
        pred = (prob > prob.max() * thresh).astype(np.uint8)
        
        pred = resize_image(pred, original_dims[1], original_dims[0])
        # prob = resize_image(prob, original_dims[1], original_dims[0])
        orig_prob = resize_image(orig_prob, original_dims[1], original_dims[0])
        
        end_time = time.time()
        if verbose:
            print(f"Time taken to predict: {end_time - start_time:.2f} seconds")
        
        if sum([has_sat for row in has_sats for has_sat in row]) == 0:
            return np.zeros((original_dims[0], original_dims[1])), orig_prob
        
        return pred, orig_prob

    
    def calc_accuracy(self, dataset: Dataset, thresh: float = THRESHOLD, verbose: bool = False):
        tp, fp, fn = 0, 0, 0
        
        accuracies = []
        ratios = []
        for idx in tqdm(range(len(dataset))):
            item = dataset[idx]
            image = np.array(item['image'])
            bbox = item['bbox']
            mask_true = np.array(item['mask'])[:, :, 0]
            
            mask_pred, _ = self.predict(image, thresh, verbose)
            
            if bbox is None:
                if mask_pred.max() == 0:
                    accuracies.append(1)
                    tp += 1
                else:
                    accuracies.append(0)
                    fp += 1
                continue
            
            # within, ratio = mask_within_bbox2(mask_pred, bbox)
            within, ratio = joint_intersection(mask_pred, mask_true, IoU_THRESHOLD)
            ratios.append(ratio)
            if within:
                accuracies.append(1)
                tp += 1
            else:
                accuracies.append(0)
                fn += 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precision_recall = {'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}
        
        return np.mean(accuracies), np.mean([ratio for ratio in ratios if ratio > IoU_THRESHOLD]), precision_recall

    # def calculate_precision_recall(self, dataset: Dataset, thresh: float = THRESHOLD, verbose: bool = False):
    #     tp, fp, fn = 0, 0, 0
    #     for idx in tqdm(range(len(dataset))):
    #         item = dataset[idx]
    #         image = np.array(item['image'])
    #         bbox = item['bbox']
    #         mask_true = np.array(item['mask'])[:, :, 0]
            
    #         mask_pred, _ = self.predict(image, thresh, verbose)
            
    #         if bbox is None:
    #             if mask_pred.max() == 0:
    #                 tp += 1
    #             else:
    #                 fp += 1
    #             continue
            
    #         within, ratio = joint_intersection(mask_pred, mask_true, IoU_THRESHOLD)
    #         if within:
    #             tp += 1
    #         else:
    #             fn += 1
        
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
        
    #     return precision, recall
    
    def sample_prediction(self, dataset: Dataset, thresh: float = THRESHOLD, idx: int = None, verbose=False, iou_or_bbox: str = 'iou'):
        idx = random.randint(0, len(dataset)-1) if not idx else idx
        
        item = dataset[idx]
        image = np.array(item['image'])
        bbox = item['bbox']
        mask = np.array(item['mask'])

        mask_pred, mask_prob = self.predict(image, thresh, verbose=verbose)

        within, ratio = False, 0
        if iou_or_bbox == 'bbox':
            within, ratio = mask_within_bbox(mask_pred, bbox)
        elif iou_or_bbox == 'iou':
            within, ratio = joint_intersection(mask_pred, mask[:, :, 0], IoU_THRESHOLD)
            
        print("IoU:", ratio)
        if within:
            print("Correctly segmented!")
        else:
            print("Incorrectly segmented.")

        plot_prediction(image, mask_pred, mask_prob, idx, bbox)
        
        return mask_pred, mask_prob