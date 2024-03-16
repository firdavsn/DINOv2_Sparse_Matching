# src/dinov2_custom/segmenter_transformers.py

import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline, SamModel, SamConfig, SamProcessor
import torch

import os

MODEL_TYPES = {"large" : "facebook/sam-vit-huge",
               "medium": "facebook/sam-vit-large",
               "small" : "facebook/sam-vit-base"}

class Segmenter_Transformers:
    """
    Segmenter class to apply semantic segmentation to images using SAM from 
    transformers module.
    """

    def __init__(self, checkpoint: str = None, model_size: str = "mobile", device: str = "cuda"):
        """
        Initialize Segmenter model and any utils.

        Arguments:
            model_size (str, "large"): Size of the model, options are 'large', 'medium', 'small' each corresponding to vit_huge, vit_large, vit_base, respectively.
            device (str, "cuda"): Device to run on; options are 'cuda' or 'cpu'.
        """
        
        device = 0 if device == 'cuda' else -1
        
        pre_trained = checkpoint is None
        
        if pre_trained:
            self.mask_generator = pipeline("mask-generation", model=MODEL_TYPES[model_size], device=device)
        else:
            model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
            model = SamModel(config=model_config)
            processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            
            model.load_state_dict(torch.load(checkpoint))
            
            self.mask_generator = pipeline("mask-generation", model=model, feature_extractor=processor, device=device)

    def generate_masks(self, image: np.ndarray) -> list[dict]:
        """
        Generates masks of the image.

        Arguments:
            image (np.ndarray): Image represented in an array.

        Returns:
            dict[str, list]: Masks representing each masked section of the image.
        """

        return self.mask_generator(image, points_per_batch=64)

    # TODO
    def is_valid_masks(self, masks: list[dict]) -> bool:
        """
        Checks if the masks list is a valid list of masks.

        Arguments:
            masks (list[dict]): List of dictionaries representing each masked section of the image.

        Returns:
            bool: Whether or not masks is a valid list.
        """

        return True

    def prepare_mask(self, mask: dict, is_bw: bool = False) -> np.ndarray:
        """
        Gets the image array of the mask.

        Arguments:
            image (np.ndarray): Image represented in an array.
            mask (dict): Dictionary representing the mask to prepare.
            is_bw (bool) : Whether or not mask overlay has black and white filter.

        Returns:
            np.ndarray: Image array of the prepared mask.
        """

        img = np.ones((mask.shape[0], mask.shape[1], 4))
        img[:,:,3] = 0

        m = mask
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

        return img

    def prepare_masks(self, masks: list[dict], is_bw: bool = False) -> np.ndarray:
        """
        Gets the image array of a list of masks.

        Arguments:
            image (np.ndarray): Image represented in an array.
            masks (list[dict]): List of dictionaries representing each masked section of the image.
            is_bw (bool) : Whether or not mask overlay has black and white filter.

        Returns:
            np.ndarray: Image array of the prepared masks.
        """

        if len(masks) == 0:
            return

        img = np.ones((masks[0].shape[0], masks[0].shape[1], 4))
        img[:,:,3] = 0

        for mask in masks:
            m = mask
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask

        return img

    def visualize(self, mask_overlay: np.ndarray, image: np.ndarray, figsize: tuple[int] = (15, 5)):
        """
        Visualizes the mask. Overlays it on the image (if any).

        Arguments:
            mask_overlay (np.ndarray): Mask overlay image array.
            image (np.ndarray): Image represented in an array.
            figsize (tuple[int]): Matplotlib figure size.

        """

        is_bw = len(mask_overlay.shape) == 2

        # Display the original image
        plt.figure(figsize=figsize)
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        if is_bw:
            # Convert the mask to have 3 channels
            filter = np.stack([mask_overlay, mask_overlay, mask_overlay], axis=-1)

            # Apply the mask to the image
            masked_image = image * filter

            # Display the masked image
            plt.subplot(1, 3, 2)
            plt.imshow(masked_image)
            plt.title('Masked Image')
            plt.axis('off')

            # Display the mask overlay
            plt.subplot(1, 3, 3)
            plt.imshow(mask_overlay, cmap="gray")
            plt.title('Mask Overlay')
            plt.axis('off')
        else:
            # Display the masked image
            plt.subplot(1, 3, 2)
            plt.imshow(image)
            ax = plt.gca()
            ax.set_autoscale_on(False)
            ax.imshow(mask_overlay)
            plt.title('Masked Image')
            plt.axis('off')

            # Display the mask overlay
            plt.subplot(1, 3, 3)
            plt.imshow(mask_overlay)
            plt.title('Mask Overlay')
            plt.axis('off')

        plt.show()

    def convert_bw(self, mask_overlay: np.ndarray) -> np.ndarray:
        """
        Generate black and white filter on maks. Make any pixel not in overlay black
        and everything else white.

        Arguments:
            mask_overlay (np.ndarray) : Mask overlay image array

        Returns:
            np.ndarray : Black and white mask overlay
        """

        bw_overlay = np.all(mask_overlay[:, :, :3] == 1, axis=-1).astype(float)
        bw_overlay = np.where(bw_overlay==1., 0., 1.)

        return bw_overlay